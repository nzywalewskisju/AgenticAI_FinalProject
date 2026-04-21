# src/agents/governor.py
# Governor Sub-Agent — runs twice per query: pre-check and post-check.
# Calls tools from tools/governance.py only.
# Pre-check (before Reasoning):
#   - Combined single LLM call for PII detection and escalation risk assessment
#   - Keyword check for always-escalate topics runs first with no LLM call needed
#   - high-stakes topics always escalate regardless of score
# Post-check (after Review passes):
#   - compliance_stamp: flag legally dangerous absolute statements in the final answer
#   - write_audit_log: append full interaction record to logs/audit_log.jsonl
# If pre-check blocks the query, no other agents are called.
# Audit logging is non-negotiable — runs after every answered query without exception.

from src.tools.governance import (
    compliance_stamp,
    write_audit_log,
    ALWAYS_ESCALATE_TOPICS
)
from src.tools.utils import call_llm, safe_json_parse
from config import ESCALATION_THRESHOLD


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
    Uses a single combined LLM call instead of two separate calls for speed.
    Keyword check for always-escalate topics runs first — no LLM needed.
    Returns {cleared: bool, reason: str, message: str, escalated: bool}
    """
    # Always-escalate keyword check — no LLM call needed
    query_lower = query.lower()
    for topic in ALWAYS_ESCALATE_TOPICS:
        if topic in query_lower:
            return {
                "cleared": False,
                "reason": f"Always-escalate topic detected: {topic}",
                "message": ESCALATION_MESSAGE,
                "escalated": True
            }

    # Single combined LLM call for PII + escalation risk
    system_prompt = f"""You are a compliance checker for an HR policy assistant.
Analyze the query and return both a PII check and an escalation risk score.

PII check: does the query contain private information about another employee
(not the user themselves)? Names combined with sensitive context, SSNs,
medical details about someone else, or another employee's salary count as PII.
It is fine for the user to mention facts about themselves.

Escalation risk: score 0.0 to 1.0.
>= {ESCALATION_THRESHOLD} = escalate to human HR.
High risk topics: harassment, discrimination, termination disputes, FMLA,
medical accommodation, retaliation, whistleblower, legal threats.
Low risk: general policy questions, PTO, benefits, expense reimbursement.

Respond only in JSON:
{{
  "contains_pii": true/false,
  "pii_reason": "explanation",
  "risk_score": 0.0,
  "risk_reason": "explanation"
}}"""

    response = call_llm(query, system_prompt=system_prompt)
    result = safe_json_parse(response, fallback={
        "contains_pii": False,
        "pii_reason": "",
        "risk_score": 0.0,
        "risk_reason": ""
    })

    if result.get("contains_pii"):
        return {
            "cleared": False,
            "reason": result.get("pii_reason", "PII detected"),
            "message": PII_MESSAGE,
            "escalated": False
        }

    risk_score = float(result.get("risk_score", 0.0))
    if risk_score >= ESCALATION_THRESHOLD:
        return {
            "cleared": False,
            "reason": result.get("risk_reason", "High escalation risk"),
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
    Audit log is always written regardless of compliance result.
    """
    compliance_result = compliance_stamp(final_answer)
    compliance_passed = compliance_result.get("passed", True)

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