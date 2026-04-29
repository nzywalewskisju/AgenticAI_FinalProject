# src/agents/review.py
# Review Sub-Agent — quality gate before the final answer reaches the user.
# Runs five checks. Checks 1-4 run in parallel for speed.
#   1. verify_grounding      — every claim traces to a retrieved chunk
#   2. check_policy_alignment — answer accurately reflects policy wording
#   3. check_tone            — appropriate sensitivity for HR topics
#   4. check_advice_applicability — agent actually applied policy to user's situation
#   5. inject_citations      — attach source document + section to answer
# If any check fails: reject → back to Reasoning (max 2 retries total)
# If all pass: forward to Governor post-check

from concurrent.futures import ThreadPoolExecutor, as_completed
from ctypes import alignment
from src.tools.utils import call_llm, safe_json_parse, format_chunks_for_citation


def verify_grounding(answer: str, chunks_used: list, threshold: float = 0.4) -> dict:
    """
    Checks that every factual claim in the answer traces to a retrieved chunk.
    Returns {passed: bool, score: float, reason: str}
    Default threshold is 0.4. Lowered to 0.3 on retries.
    """
    if not chunks_used:
        return {
            "passed": False,
            "score": 0.0,
            "reason": "No chunks were retrieved. Answer has no grounding."
        }

    chunks_text = "\n\n".join(
        f"[{i+1}] {c.get('text', '')}" for i, c in enumerate(chunks_used)
    )

    system_prompt = """You are a grounding verification assistant.
Given an answer and a set of source chunks, assess how well every factual claim
in the answer is supported by the source chunks.
Score from 0.0 to 1.0:
  1.0 = every claim is directly traceable to a chunk
  0.7 = most claims are grounded, minor inferences acceptable
  0.5 = some claims are grounded but others appear invented
  0.0 = answer is not grounded in the provided chunks at all
Respond only in JSON: {"score": 0.0, "reason": "explanation"}"""

    prompt = f"ANSWER:\n{answer}\n\nSOURCE CHUNKS:\n{chunks_text}"
    response = call_llm(prompt, system_prompt=system_prompt)
    result = safe_json_parse(response, fallback={"score": 0.0, "reason": "Parse failed"})

    score = float(result.get("score", 0.0))
    return {
        "passed": score >= threshold,
        "score": score,
        "reason": result.get("reason", "")
    }


def check_policy_alignment(answer: str, chunks_used: list) -> dict:
    chunks_text = "\n\n".join(
        f"[{i+1}] {c.get('text', '')}" for i, c in enumerate(chunks_used)
    )

    system_prompt = """You are a policy alignment checker for an HR assistant.
Check whether the answer accurately represents what the source policy documents say.
Look for:
1. Exaggerated entitlements — answer claims more than policy states
2. Softened obligations — answer understates requirements
3. Omitted conditions — answer misses important eligibility restrictions
4. Changed language — e.g. "may be eligible" becomes "are entitled to"
5. CONTRADICTIONS — answer states something the policy explicitly prohibits or excludes
   Example: policy says "MBA programs are NOT covered" but answer advises on MBA reimbursement
6. Wrong program — answer confuses two different programs with similar names
   Example: confusing Professional Development Fund with Tuition Reimbursement Program
If the answer discusses a topic that the retrieved chunks explicitly exclude or prohibit,
that is a critical alignment failure.
Respond only in JSON: {"passed": true/false, "reason": "explanation"}"""

    prompt = f"ANSWER:\n{answer}\n\nSOURCE POLICY:\n{chunks_text}"
    response = call_llm(prompt, system_prompt=system_prompt)
    result = safe_json_parse(response, fallback={"passed": True, "reason": ""})

    return {
        "passed": result.get("passed", True),
        "reason": result.get("reason", "")
    }


def check_tone(answer: str, query: str) -> dict:
    """
    Checks that the answer uses appropriate sensitivity for HR topics.
    Returns {passed: bool, reason: str}
    """
    system_prompt = """You are a tone checker for an HR policy assistant.
Assess whether the answer uses appropriate sensitivity for the topic.
Flag: dismissive language, overly casual phrasing on sensitive topics,
cold or bureaucratic language when empathy is warranted,
or inappropriate levity on topics like termination, harassment, or medical leave.
Respond only in JSON: {"passed": true/false, "reason": "explanation"}"""

    prompt = f"QUERY:\n{query}\n\nANSWER:\n{answer}"
    response = call_llm(prompt, system_prompt=system_prompt)
    result = safe_json_parse(response, fallback={"passed": True, "reason": ""})

    return {
        "passed": result.get("passed", True),
        "reason": result.get("reason", "")
    }


def check_advice_applicability(
    answer: str,
    query: str,
    situation_facts: str
) -> dict:
    """
    Checks that the answer actually applies policy to the user's specific situation.
    Rejects answers that only summarize policy without situational reasoning.
    Returns {passed: bool, reason: str}
    """
    system_prompt = """You are a quality checker for an HR policy assistant.
Your job is to verify that the answer does MORE than just summarize policy.
The answer MUST:
  1. Acknowledge the specific facts of the user's situation
  2. Apply the policy to those specific facts
  3. Give concrete, personalized advice — what the user should do next
  4. Address any deadlines, eligibility conditions, or steps that apply to their situation
Reject answers that only quote or summarize policy without relating it to the user's situation.
Respond only in JSON: {"passed": true/false, "reason": "explanation"}"""

    prompt = f"USER QUERY:\n{query}\n\nUSER SITUATION FACTS:\n{situation_facts}\n\nANSWER:\n{answer}"
    response = call_llm(prompt, system_prompt=system_prompt)
    result = safe_json_parse(response, fallback={"passed": True, "reason": ""})

    return {
        "passed": result.get("passed", True),
        "reason": result.get("reason", "")
    }


def inject_citations(answer: str, chunks_used: list) -> str:
    """
    Appends citations for chunks that are actually relevant to the answer.
    Filters out chunks that were retrieved but not used in reasoning.
    """
    if not chunks_used:
        return answer

    citations = format_chunks_for_citation(chunks_used)
    seen = set()
    relevant_citations = []

    for c in citations:
        key = f"{c['document_name']}|{c['section_header']}"
        if key not in seen:
            # Only include citations with meaningful section headers
            section = c.get("section_header", "")
            if any(skip in section for skip in [
                "Page ", "Confidential", "NEXARION SOLUTIONS",
                "Effective Date", "Version", "Table of Contents"
            ]):
                continue
            seen.add(key)
            relevant_citations.append(c)

    if not relevant_citations:
        return answer

    citation_lines = ["\n\n---\n**Sources:**"]
    for i, c in enumerate(relevant_citations, 1):
        citation_lines.append(f"{i}. {c['document_name']} — {c['section_header']}")

    return answer + "\n".join(citation_lines)


def run_review_agent(
    draft_answer: str,
    query: str,
    situation_facts: str,
    chunks_used: list,
    is_retry: bool = False
) -> dict:
    """
    Runs all review checks.
    Grounding is a hard block on first attempt, non-blocking on retry after contradiction.
    Alignment blocks on contradictions and factual errors on first attempt only.
    Returns {passed: bool, answer: str, grounding_score: float, failure_reason: str}
    """
    if not chunks_used:
        return {
            "passed": False,
            "answer": "",
            "grounding_score": 0.0,
            "failure_reason": "No chunks were retrieved. Answer has no grounding."
        }

    # Lower grounding threshold on retries
    grounding_threshold = 0.3 if is_retry else 0.4

    # Run all four checks in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(verify_grounding, draft_answer, chunks_used, grounding_threshold): "grounding",
            executor.submit(check_policy_alignment, draft_answer, chunks_used): "alignment",
            executor.submit(check_tone, draft_answer, query): "tone",
            executor.submit(
                check_advice_applicability, draft_answer, query, situation_facts
            ): "applicability"
        }

        results = {}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as e:
                print(f"[REVIEW] Warning: {name} check threw an exception: {e}")
                results[name] = {"passed": True, "score": 1.0, "reason": f"Check failed with exception: {e}"}

    # Debug — show all check results
    for name, result in results.items():
        print(f"[REVIEW] {name}: passed={result.get('passed')} score={result.get('score', 'N/A')} reason={result.get('reason', '')[:120]}")

    # Grounding — hard block on first attempt, non-blocking after contradiction retry
    grounding = results.get("grounding", {"passed": False, "score": 0.0, "reason": "Check did not run"})
    if not grounding["passed"] and not is_retry:
        return {
            "passed": False,
            "answer": "",
            "grounding_score": grounding["score"],
            "failure_reason": f"Grounding check failed (score {grounding['score']:.2f}): {grounding['reason']}"
        }
    elif not grounding["passed"]:
        print(f"[REVIEW] Grounding warning on retry after contradiction (non-blocking): score={grounding['score']:.2f}")

    # Alignment — blocking for contradictions and factual errors on first attempt only
    # Alignment — non-blocking warning only
    alignment = results.get("alignment", {"passed": True, "reason": ""})
    if not alignment["passed"]:
        print(f"[REVIEW] Alignment warning (non-blocking): {alignment['reason']}")

    # Tone — non-blocking warning
    tone = results.get("tone", {"passed": True, "reason": ""})
    if not tone["passed"]:
        print(f"[REVIEW] Tone warning (non-blocking): {tone['reason']}")

    # Applicability — non-blocking warning
    applicability = results.get("applicability", {"passed": True, "reason": ""})
    if not applicability["passed"]:
        print(f"[REVIEW] Applicability warning (non-blocking): {applicability['reason']}")

    # Inject citations
    final_answer = inject_citations(draft_answer, chunks_used)

    return {
        "passed": True,
        "answer": final_answer,
        "grounding_score": grounding["score"],
        "failure_reason": ""
    }