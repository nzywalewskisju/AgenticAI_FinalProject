# orchestrator.py
# Central coordinator for the PolicyPro HR assistant.
# Receives every user query, runs the governor pre-check, manages the
# reasoning and review retry loop, and assembles the final answer.
# Handles out-of-scope routing, document availability checks, session
# and profile context injection, and date augmentation.
#
# Functions: run_orchestrator, classify_query, _is_followup_query

import uuid
import queue
from pydantic import BaseModel
from config import ROUTE_IN_SCOPE, ROUTE_HIGH_STAKES, ROUTE_OUT_OF_SCOPE
from src.tools.utils import call_llm, get_current_date, safe_json_parse
from src.memory.session import session_memory
from src.memory.profile import extract_and_update_profile, get_profile_context_string, get_relevant_profile_context, load_profile
from src.memory.registry import has_documents
from src.agents.governor import run_governance_precheck, run_governance_postcheck
from src.agents.reasoning import run_reasoning_agent
from src.agents.review import run_review_agent


MAX_REVIEW_RETRIES = 3


class RoutingDecision(BaseModel):
    category: str       # ROUTE_IN_SCOPE | ROUTE_HIGH_STAKES | ROUTE_OUT_OF_SCOPE
    confidence: float
    reasoning: str


def classify_query(query: str) -> RoutingDecision:
    # Sends the query to the LLM to determine if it is HR-related or
    # out of scope. Returns a routing object with a category field.
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
  - Technology policies, acceptable use of company systems
  - AI tools policy, approved software, prohibited software
  - Data handling, confidentiality, acceptable use of technology
  - Any question about what a specific company tool or software policy says
  - Any question about whether a specific action is permitted under company policy
  - Any question about what approval process is required for a workplace action
  - Any question about what an employee should use instead of a prohibited tool
  - Any question about what a company policy says or covers

When in doubt, classify as "{ROUTE_IN_SCOPE}". It is always better to attempt an HR answer than to incorrectly reject a legitimate policy question.

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

def _is_followup_query(query: str, session_context: str) -> bool:
    # Returns True if the query contains signals that it refers to a
    # previous question, such as "what about" or "and also".
    if not session_context or "No prior" in session_context:
        return False
    followup_signals = [
        "what about", "and also", "what if", "how about",
        "same question", "follow up", "additionally",
        "also", "another question", "one more", "what else",
        "in that case", "given that", "based on that",
        "can you clarify", "you mentioned", "you said",
        "it", "that", "this", "those", "they", "them"
    ]
    query_lower = query.lower()
    return any(signal in query_lower for signal in followup_signals)

def run_orchestrator(
    query: str,
    user_id: str,
    session_id: str = None,
    status_queue: queue.Queue = None
) -> dict:
    # Entry point for every user query. Runs pre-check, manages the
    # reasoning and review retry loop, and returns the final answer.

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
    profile_context = get_relevant_profile_context(user_id, query)

    # Only inject session history if this appears to be a follow-up question
    if not _is_followup_query(query, session_context):
        session_context = ""

    from src.tools.utils import get_current_date
    current_date = get_current_date()
    augmented_query = f"{query}\n\nToday's date: {current_date}. All policy questions refer to the current year."

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
            query=augmented_query + retry_context,
            user_id=user_id,
            session_context=session_context,
            profile_context=profile_context,
            prior_chunks=all_chunks_accumulated,
            status_queue=status_queue
        )

        # Accumulate chunks across retries
        accumulated_texts = {c["text"] for c in all_chunks_accumulated}
        for c in reasoning_result.get("chunks_used", []):
            if c["text"] not in accumulated_texts:
                all_chunks_accumulated.append(c)
                accumulated_texts.add(c["text"])

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
        failure_reason = review_result["failure_reason"] if review_result else ""
        if "contradiction" in failure_reason.lower():
            no_info_answer = (
                "I was unable to produce a verified answer for this question. "
                "The policy documents contain specific rules that my initial answer did not correctly reflect. "
                "Please contact HR directly for accurate guidance on this topic."
            )
        else:
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