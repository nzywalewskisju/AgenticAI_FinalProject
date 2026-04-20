# src/agents/governor.py
# Governor Sub-Agent — runs twice per query: pre-check and post-check.
# Calls tools from tools/governance.py only.
# Pre-check (before Reasoning):
#   - detect_pii: block queries containing another employee's private information
#   - assess_escalation_risk: score 0-1, escalate to human HR if >= ESCALATION_THRESHOLD
#   - high-stakes topics always escalate regardless of score
# Post-check (after Review passes):
#   - compliance_stamp: flag legally dangerous absolute statements in the final answer
#   - write_audit_log: append full interaction record to logs/audit_log.jsonl
# If pre-check blocks the query, no other agents are called.
# Audit logging is non-negotiable — runs after every answered query without exception.

from src.tools.governance import (
    detect_pii,
    assess_escalation_risk,
    compliance_stamp,
    write_audit_log
)


ESCALATION_MESSAGE = (
    "Your query has been flagged as sensitive and requires direct HR involvement. "
    "Please contact HR directly to discuss this matter. "
    "If this is urgent, please reach out to your HR representative immediately."
)

PII_MESSAGE = (
    "Your query appears to contain private information about another employee. "
    "This assistant can only help with questions about your own employment situation. "
    "Please rephrase your question without referencing another employee's personal details."
)


def run_governance_precheck(query: str, user_id: str) -> dict:
    """
    Runs PII detection and escalation risk assessment before any reasoning occurs.
    Returns {cleared: bool, reason: str, message: str, escalated: bool}
    If cleared is False, the orchestrator stops and returns message to the user.
    """
    # PII check first — if another employee's data is in the query, stop immediately
    pii_result = detect_pii(query)
    if pii_result.get("contains_pii"):
        return {
            "cleared": False,
            "reason": f"PII detected: {pii_result.get('reason', '')}",
            "message": PII_MESSAGE,
            "escalated": False
        }

    # Escalation risk check
    escalation_result = assess_escalation_risk(query)
    if escalation_result.get("should_escalate"):
        return {
            "cleared": False,
            "reason": f"Escalation required: {escalation_result.get('reason', '')}",
            "message": ESCALATION_MESSAGE,
            "escalated": True
        }

    return {
        "cleared": True,
        "reason": "Query cleared for processing.",
        "message": "",
        "escalated": False
    }


def run_governance_postcheck(
    session_id: str,
    user_id: str,
    query: str,
    route: str,
    chunks_used: list,
    situation_facts: str,
    final_answer: str,
    grounding_score: float
) -> dict:
    """
    Runs compliance stamp and audit logging after the answer passes Review.
    Returns {passed: bool, flagged_phrases: list, answer: str}
    The answer may be modified to add a disclaimer if compliance issues are found.
    Audit log is always written regardless of compliance result.
    """
    compliance_result = compliance_stamp(final_answer)
    compliance_passed = compliance_result.get("passed", True)

    # If compliance fails, append a disclaimer rather than rejecting entirely
    answer_out = final_answer
    if not compliance_passed:
        flagged = compliance_result.get("flagged_phrases", [])
        answer_out += (
            "\n\n---\n"
            "_Note: This response is for informational purposes only and does not "
            "constitute legal advice. Please consult with HR or a qualified professional "
            "for guidance on your specific situation._"
        )
        print(f"[GOVERNOR] Compliance warning — flagged phrases: {flagged}")

    # Audit log — always runs, never raises
    write_audit_log(
        session_id=session_id,
        user_id=user_id,
        query=query,
        route=route,
        escalated=False,
        chunks_used=chunks_used,
        situation_facts=situation_facts,
        final_answer=answer_out,
        grounding_score=grounding_score,
        compliance_passed=compliance_passed
    )

    return {
        "passed": compliance_passed,
        "flagged_phrases": compliance_result.get("flagged_phrases", []),
        "answer": answer_out
    }