# src/agents/orchestrator.py
# Top-level orchestrator — main entry point for all queries.
# Responsibilities:
#   - classify_query: route to hr_in_scope, high_stakes, or out_of_scope
#   - manage_session_memory: attach prior conversation turns to the query context
#   - extract_profile_facts: detect personal facts in the query and persist to profile
#   - check_document_availability: if no documents ingested, ask for a file
#   - coordinate the full agent pipeline in order:
#       Governor pre-check → Reasoning → Review → Governor post-check
#   - enforce no-chunks guard: reject any answer produced without retrieved evidence
#   - enforce max 2 retries if Review rejects the Reasoning output
#   - return final answer or escalation/clarification message to the GUI

import uuid
import queue
from pydantic import BaseModel
from config import ROUTE_IN_SCOPE, ROUTE_HIGH_STAKES, ROUTE_OUT_OF_SCOPE
from src.tools.utils import call_llm, safe_json_parse
from src.memory.session import session_memory
from src.memory.profile import extract_and_update_profile, get_profile_context_string
from src.memory.registry import has_documents
from src.agents.governor import run_governance_precheck, run_governance_postcheck
from src.agents.reasoning import run_reasoning_agent
from src.agents.review import run_review_agent


MAX_REVIEW_RETRIES = 2


class RoutingDecision(BaseModel):
    category: str       # ROUTE_IN_SCOPE | ROUTE_HIGH_STAKES | ROUTE_OUT_OF_SCOPE
    confidence: float
    reasoning: str


def classify_query(query: str) -> RoutingDecision:
    """
    Classifies the query into one of three routing categories.
    Uses structured JSON output validated with Pydantic.
    """
    system_prompt = f"""You are a query classifier for an HR Policy Assistant.
Classify the query into exactly one of these categories:

"{ROUTE_IN_SCOPE}": ANY question about employment, workplace, or company policies including:
  - Health insurance, medical plans, dental, vision
  - FSA, HSA, healthcare savings accounts, rollover amounts, contribution limits
  - COBRA, benefits continuation after leaving
  - 401k, retirement, pension, matching contributions
  - PTO, vacation, sick leave, holidays, time off
  - Payroll, salary, pay schedule, compensation, bonuses
  - Remote work, hybrid work, work from home policies
  - Expense reimbursement, travel policy
  - Onboarding, offboarding, resignation, termination
  - Parental leave, FMLA, medical leave, disability leave
  - Performance reviews, promotions, PIPs
  - Dress code, conduct, workplace policies
  - Benefits enrollment, open enrollment, qualifying life events
  - Wellness stipend, professional development, EAP
  - Any question about what a company policy says or covers

"{ROUTE_HIGH_STAKES}": Sensitive HR matters requiring human involvement:
  - Harassment, discrimination, hostile work environment
  - Termination disputes, wrongful termination
  - FMLA disputes, medical accommodation requests, ADA
  - Retaliation claims, whistleblower complaints
  - Legal threats against the company or employee

"{ROUTE_OUT_OF_SCOPE}": ONLY questions with zero connection to employment or workplace:
  - Coding help, math problems, science questions
  - Weather, sports, entertainment, news
  - Personal life advice unrelated to work
  - General knowledge questions about the world

When in doubt, classify as "{ROUTE_IN_SCOPE}". It is always better to attempt an HR answer than to incorrectly reject a legitimate benefits question.

Respond only in JSON: {{"category": "...", "confidence": 0.0, "reasoning": "..."}}"""

    response = call_llm(query, system_prompt=system_prompt)
    result = safe_json_parse(response, fallback={
        "category": ROUTE_IN_SCOPE,
        "confidence": 0.5,
        "reasoning": "Classification parse failed — defaulting to in_scope"
    })

    return RoutingDecision(
        category=result.get("category", ROUTE_IN_SCOPE),
        confidence=float(result.get("confidence", 0.5)),
        reasoning=result.get("reasoning", "")
    )


def run_orchestrator(
    query: str,
    user_id: str,
    session_id: str = None,
    status_queue: queue.Queue = None
) -> dict:
    if session_id is None:
        session_id = str(uuid.uuid4())

    loading_steps = []

    def step(msg: str):
        loading_steps.append(msg)
        print(f"[ORCHESTRATOR] {msg}")
        if status_queue:
            status_queue.put(msg)

    # ── Step 1: Extract profile facts ─────────────────────────────────────────
    step("Checking your profile...")
    new_facts = extract_and_update_profile(user_id, query)

    # ── Step 2: Classify the query ─────────────────────────────────────────────
    step("Classifying your query...")
    routing = classify_query(query)

    # ── Step 3: Handle out-of-scope ────────────────────────────────────────────
    if routing.category == ROUTE_OUT_OF_SCOPE:
        return {
            "answer": "I can only help with HR policy questions. Your question appears to be outside that scope. Please ask about topics like PTO, benefits, workplace policies, or your employment situation.",
            "status": "out_of_scope",
            "route": ROUTE_OUT_OF_SCOPE,
            "new_profile_facts": new_facts,
            "chunks_used": [],
            "grounding_score": 0.0,
            "loading_steps": loading_steps
        }

    # ── Step 4: Check document availability ───────────────────────────────────
    step("Checking available documents...")
    if not has_documents(user_id):
        return {
            "answer": "I don't have any HR documents loaded for your account yet. Please upload at least one HR policy document using the file picker before asking questions.",
            "status": "no_documents",
            "route": routing.category,
            "new_profile_facts": new_facts,
            "chunks_used": [],
            "grounding_score": 0.0,
            "loading_steps": loading_steps
        }

    # ── Step 5: Governor pre-check ─────────────────────────────────────────────
    step("Running compliance pre-check...")
    precheck = run_governance_precheck(query, user_id)
    if not precheck["cleared"]:
        from src.tools.governance import write_audit_log
        write_audit_log(
            session_id=session_id,
            user_id=user_id,
            query=query,
            route=routing.category,
            escalated=precheck["escalated"],
            chunks_used=[],
            situation_facts="",
            final_answer=precheck["message"],
            grounding_score=0.0,
            compliance_passed=True
        )
        return {
            "answer": precheck["message"],
            "status": "escalated" if precheck["escalated"] else "blocked",
            "route": routing.category,
            "new_profile_facts": new_facts,
            "chunks_used": [],
            "grounding_score": 0.0,
            "loading_steps": loading_steps
        }

    # ── Step 6: Get session and profile context ────────────────────────────────
    session_context = session_memory.get_context_string(session_id)
    profile_context = get_profile_context_string(user_id)

    # ── Step 7: Reasoning + Review loop ───────────────────────────────────────
    retry_count = 0
    review_result = None
    reasoning_result = None
    failure_history = []
    all_chunks_accumulated = []

    while retry_count < MAX_REVIEW_RETRIES:
        if retry_count == 0:
            step("Reasoning about your situation...")
        else:
            step(f"Refining answer (attempt {retry_count + 1})...")

        retry_context = ""
        if failure_history:
            retry_context = "\n\nPREVIOUS ATTEMPT FEEDBACK:\n" + "\n".join(failure_history)

        reasoning_result = run_reasoning_agent(
            query=query + retry_context,
            user_id=user_id,
            session_context=session_context,
            profile_context=profile_context,
            prior_chunks=all_chunks_accumulated,
            status_queue=status_queue
        )

        # Accumulate chunks across retries
        for c in reasoning_result.get("chunks_used", []):
            if c not in all_chunks_accumulated:
                all_chunks_accumulated.append(c)

        print(f"[ORCHESTRATOR] Reasoning status: {reasoning_result['status']}")
        print(f"[ORCHESTRATOR] Chunks used: {len(reasoning_result['chunks_used'])}")
        print(f"[ORCHESTRATOR] Iterations: {reasoning_result['iterations']}")

        # Guard 1 — handle clarification
        if reasoning_result["status"] == "clarification":
            return {
                "answer": reasoning_result["draft_answer"],
                "status": "clarification",
                "route": routing.category,
                "new_profile_facts": new_facts,
                "chunks_used": [],
                "grounding_score": 0.0,
                "loading_steps": loading_steps
            }

        # Guard 2 — handle no_info (max turns reached without answer)
        if reasoning_result["status"] == "no_info":
            failure_history.append(
                "Reasoning agent reached max turns without producing an answer. "
                "Try retrieving different chunks and producing a complete Answer."
            )
            retry_count += 1
            continue

        # Guard 3 — no chunks retrieved
        if not all_chunks_accumulated:
            failure_history.append(
                "No policy chunks were retrieved. "
                "You must retrieve relevant chunks before answering."
            )
            retry_count += 1
            continue

        # Guard 4 — empty draft answer
        if not reasoning_result.get("draft_answer", "").strip():
            print(f"[ORCHESTRATOR] Warning: draft_answer is empty — skipping review")
            failure_history.append(
                "Draft answer was empty. "
                "You must produce a complete Answer statement before finishing."
            )
            retry_count += 1
            continue

        # Run Review
        step("Reviewing answer quality...")
        review_result = run_review_agent(
            draft_answer=reasoning_result["draft_answer"],
            query=query,
            situation_facts=reasoning_result["situation_facts"],
            chunks_used=all_chunks_accumulated,
            is_retry=retry_count > 0
        )

        if review_result["passed"]:
            break

        failure_history.append(f"Review rejection: {review_result['failure_reason']}")
        retry_count += 1

    # ── Step 8: Handle failure after all retries ───────────────────────────────
    if not review_result or not review_result["passed"]:
        no_info_answer = (
            "I was unable to find sufficient policy information to give you a reliable answer on this topic. "
            "This may mean the relevant policy is not in the documents you've uploaded. "
            "Would you like to upload additional HR documents that might cover this topic?"
        )
        return {
            "answer": no_info_answer,
            "status": "no_info",
            "route": routing.category,
            "new_profile_facts": new_facts,
            "chunks_used": all_chunks_accumulated,
            "grounding_score": review_result["grounding_score"] if review_result else 0.0,
            "loading_steps": loading_steps
        }

    # ── Step 9: Governor post-check ────────────────────────────────────────────
    step("Running final compliance check...")
    postcheck = run_governance_postcheck(
        session_id=session_id,
        user_id=user_id,
        query=query,
        route=routing.category,
        chunks_used=all_chunks_accumulated,
        situation_facts=reasoning_result["situation_facts"],
        final_answer=review_result["answer"],
        grounding_score=review_result["grounding_score"]
    )

    # ── Step 10: Update session memory ────────────────────────────────────────
    session_memory.add_turn(session_id, query, postcheck["answer"])

    step("Done.")

    return {
        "answer": postcheck["answer"],
        "status": "success",
        "route": routing.category,
        "new_profile_facts": new_facts,
        "chunks_used": all_chunks_accumulated,
        "grounding_score": review_result["grounding_score"],
        "loading_steps": loading_steps
    }