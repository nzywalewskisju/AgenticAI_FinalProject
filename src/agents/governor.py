# governor.py
# Security and compliance agent that runs before and after every query.
# Pre-check detects prompt injection, PII, and high-risk escalation topics
# across four layers: injection detection, always-escalate keywords,
# policy question whitelist, and combined PII and risk LLM call.
# Post-check stamps the final answer for legally dangerous language and
# writes every interaction to the audit log. Sends email alerts on
# security or escalation events.
#
# Functions: run_governance_precheck, run_governance_postcheck

from src.tools.governance import (
    compliance_stamp,
    write_audit_log,
    ALWAYS_ESCALATE_TOPICS,
    detect_prompt_injection
)
from src.tools.utils import call_llm, safe_json_parse
from config import ESCALATION_THRESHOLD
from src.tools.email_alert import send_alert_email


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
    # Runs four security layers before any reasoning begins. Blocks the
    # query and sends an email alert if injection, PII, or escalation is detected.

    # ── Layer 1: Hard keyword injection check — no LLM needed ─────────────────
    injection_result = detect_prompt_injection(query)
    print(f"[GOVERNOR] Injection check: is_injection={injection_result['is_injection']} reason={injection_result.get('reason', '')[:100]}")
    if injection_result.get("is_injection"):
        send_alert_email(
            subject="Security Alert — Prompt Injection Attempt",
            username=user_id,
            query=query,
            reason=injection_result.get("reason", "Prompt injection detected"),
            alert_type="security"
        )
        return {
            "cleared": False,
            "reason": f"Prompt injection detected: {injection_result.get('reason', '')}",
            "message": (
                "This request has been flagged as a potential security violation. "
                "This assistant is designed to answer HR policy questions only. "
                "Attempts to extract system data or manipulate the assistant are logged."
            ),
            "escalated": False
        }

    # ── Layer 2: Always-escalate keyword check — no LLM needed ────────────────
    query_lower = query.lower()
    for topic in ALWAYS_ESCALATE_TOPICS:
        if topic in query_lower:
            send_alert_email(
                subject="HR Escalation",
                username=user_id,
                query=query,
                reason=f"Always-escalate topic detected: {topic}",
                alert_type="escalation"
            )
            return {
                "cleared": False,
                "reason": f"Always-escalate topic detected: {topic}",
                "message": ESCALATION_MESSAGE,
                "escalated": True
            }

    # ── Layer 3: Whitelist — pure policy questions cleared immediately ─────────
    POLICY_QUESTION_SIGNALS = [
        "how much does", "what is the", "what are the", "how many",
        "when does", "when is", "what does", "is there a",
        "do i get", "am i eligible", "what is nexarion",
        "how does", "what happens to", "can i", "what is the limit",
        "what is the amount", "how much is", "what is the rate",
        "contribution", "rollover", "deductible", "premium", "coverage",
        "hsa", "fsa", "401k", "pto", "benefits", "enrollment",
        "cobra", "fmla", "leave", "reimbursement", "stipend",
        "professional development", "wellness", "salary", "payroll"
    ]
    if any(signal in query_lower for signal in POLICY_QUESTION_SIGNALS):
        return {
            "cleared": True,
            "reason": "Query identified as standard policy question — cleared without LLM check.",
            "message": "",
            "escalated": False
        }

    # ── Layer 4: Combined PII + escalation risk — single LLM call ─────────────
    system_prompt = f"""You are a compliance checker for an HR policy assistant.
Analyze the query and return both a PII check and an escalation risk score.

PII check: Flag ONLY if the query contains a SPECIFIC NAMED PERSON (not the user themselves)
combined with sensitive data about that person.
These are NOT PII and must NEVER be flagged:
- Questions about company benefit amounts, plan names, or contribution limits
- Questions about eligibility criteria or policy rules
- Questions that mention dollar amounts, plan names, or policy sections
- Any question about what a company policy says
Only flag PII if a real person's name appears alongside their salary, medical info, or credentials.

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
        send_alert_email(
            subject="Security Alert — PII Detected",
            username=user_id,
            query=query,
            reason=result.get("pii_reason", "PII detected in query"),
            alert_type="security"
        )
        return {
            "cleared": False,
            "reason": result.get("pii_reason", "PII detected"),
            "message": PII_MESSAGE,
            "escalated": False
        }

    risk_score = float(result.get("risk_score", 0.0))
    if risk_score >= ESCALATION_THRESHOLD:
        send_alert_email(
            subject="HR Escalation — High Risk Query",
            username=user_id,
            query=query,
            reason=result.get("risk_reason", "High escalation risk"),
            alert_type="escalation"
        )
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
    # Runs compliance stamp on the final answer and writes the full
    # interaction to the audit log. Appends a disclaimer if sensitive
    # legal language is detected.
    
    # Guard against empty answer reaching compliance stamp
    if not final_answer or not final_answer.strip():
        print(f"[GOVERNOR] Warning: empty answer received — skipping compliance stamp")
        write_audit_log(
            session_id=session_id,
            user_id=user_id,
            query=query,
            route=route,
            escalated=False,
            chunks_used=chunks_used,
            situation_facts=situation_facts,
            final_answer="",
            grounding_score=grounding_score,
            compliance_passed=False
        )
        return {
            "passed": False,
            "flagged_phrases": [],
            "answer": ""
        }

    compliance_result = compliance_stamp(final_answer)
    compliance_passed = compliance_result.get("passed", True)

    answer_out = final_answer
    if not compliance_passed:
        flagged = compliance_result.get("flagged_phrases", [])
        sensitive_topics = [
            "termination", "discrimination", "harassment", "fmla",
            "medical", "disability", "legal", "lawsuit", "attorney",
            "entitled", "guaranteed", "must by law", "required by law"
        ]
        answer_lower = final_answer.lower()
        is_sensitive = any(topic in answer_lower for topic in sensitive_topics)
        if is_sensitive:
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
