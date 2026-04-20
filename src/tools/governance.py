# src/tools/governance.py
# Governance tools owned by the Governor Sub-Agent (agents/governor.py).
# Functions:
#   detect_pii(query)
#     — scans the user's query for another employee's private information
#     — flags names + sensitive context, social security numbers, medical details
#       that refer to someone other than the querying user
#   assess_escalation_risk(query, user_id)
#     — scores the query 0.0 to 1.0 for escalation risk
#     — scores >= ESCALATION_THRESHOLD route to human HR instead of the agent
#   compliance_stamp(answer)
#     — scans the final answer for legally dangerous absolute statements
#     — flags language like "you are entitled to" or "the company must"
#     — returns passed: True/False and a list of flagged phrases
#   write_audit_log(session_id, user_id, query, route, chunks_used,
#                   final_answer, grounding_score, compliance_passed)
#     — appends a full interaction record to logs/audit_log.jsonl
#     — called after every answered query, non-negotiable