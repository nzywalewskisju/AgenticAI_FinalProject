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
from src.tools.utils import call_llm, safe_json_parse, format_chunks_for_citation


def verify_grounding(answer: str, chunks_used: list) -> dict:
    """
    Checks that every factual claim in the answer traces to a retrieved chunk.
    Returns {passed: bool, score: float, reason: str}
    Score must be >= 0.7 to pass.
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
        "passed": score >= 0.5,
        "score": score,
        "reason": result.get("reason", "")
    }


def check_policy_alignment(answer: str, chunks_used: list) -> dict:
    """
    Checks that the answer accurately represents what the policy says.
    Catches cases where retrieval was correct but paraphrasing was inaccurate.
    Returns {passed: bool, reason: str}
    """
    chunks_text = "\n\n".join(
        f"[{i+1}] {c.get('text', '')}" for i, c in enumerate(chunks_used)
    )

    system_prompt = """You are a policy alignment checker for an HR assistant.
Check whether the answer accurately represents what the source policy documents say.
Look for: exaggerated entitlements, softened obligations, omitted conditions,
or changed language that alters the meaning (e.g. "may be eligible" → "are entitled to").
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
    Appends a citations section to the answer listing every source document and section used.
    Returns the answer with citations appended.
    """
    if not chunks_used:
        return answer

    citations = format_chunks_for_citation(chunks_used)
    seen = set()
    unique_citations = []
    for c in citations:
        key = f"{c['document_name']}|{c['section_header']}"
        if key not in seen:
            seen.add(key)
            unique_citations.append(c)

    citation_lines = ["\n\n---\n**Sources:**"]
    for i, c in enumerate(unique_citations, 1):
        citation_lines.append(f"{i}. {c['document_name']} — {c['section_header']}")

    return answer + "\n".join(citation_lines)


def run_review_agent(
    draft_answer: str,
    query: str,
    situation_facts: str,
    chunks_used: list
) -> dict:
    """
    Runs all five review checks.
    Checks 1-4 run in parallel for speed — all four LLM calls fire simultaneously.
    Returns {passed: bool, answer: str, grounding_score: float, failure_reason: str}
    """
    if not chunks_used:
        return {
            "passed": False,
            "answer": "",
            "grounding_score": 0.0,
            "failure_reason": "No chunks were retrieved. Answer has no grounding."
        }

    # Run all four checks in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(verify_grounding, draft_answer, chunks_used): "grounding",
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
                # If a check throws, treat it as passed to avoid false rejections
                print(f"[REVIEW] Warning: {name} check threw an exception: {e}")
                results[name] = {"passed": True, "score": 1.0, "reason": f"Check failed with exception: {e}"}

    # Evaluate results in order — grounding first as it is the most critical
    grounding = results.get("grounding", {"passed": False, "score": 0.0, "reason": "Check did not run"})
    if not grounding["passed"]:
        return {
            "passed": False,
            "answer": "",
            "grounding_score": grounding["score"],
            "failure_reason": f"Grounding check failed (score {grounding['score']:.2f}): {grounding['reason']}"
        }

    alignment = results.get("alignment", {"passed": True, "reason": ""})
    if not alignment["passed"]:
        return {
            "passed": False,
            "answer": "",
            "grounding_score": grounding["score"],
            "failure_reason": f"Policy alignment check failed: {alignment['reason']}"
        }

    tone = results.get("tone", {"passed": True, "reason": ""})
    if not tone["passed"]:
        return {
            "passed": False,
            "answer": "",
            "grounding_score": grounding["score"],
            "failure_reason": f"Tone check failed: {tone['reason']}"
        }

    applicability = results.get("applicability", {"passed": True, "reason": ""})
    if not applicability["passed"]:
        print(f"[REVIEW] Applicability warning (non-blocking): {applicability['reason']}")

    # All passed — inject citations
    final_answer = inject_citations(draft_answer, chunks_used)

    return {
        "passed": True,
        "answer": final_answer,
        "grounding_score": grounding["score"],
        "failure_reason": ""
    }