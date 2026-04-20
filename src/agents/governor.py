# src/agents/governor.py
# Governor Sub-Agent — runs twice per query: pre-check and post-check.
# Calls tools from tools/governance.py only.
# Pre-check (before Reasoning):
#   - detect_pii: block queries containing another employee's private information
#   - assess_escalation_risk: score 0-1, escalate to human HR if >= ESCALATION_THRESHOLD
#   - high-stakes topics (harassment, discrimination, termination, FMLA, retaliation,
#     medical accommodation, whistleblower) always escalate regardless of score
# Post-check (after Review passes):
#   - compliance_stamp: flag legally dangerous absolute statements in the final answer
#   - write_audit_log: append full interaction record to logs/audit_log.jsonl
# If pre-check blocks the query, no other agents are called.
# Audit logging is non-negotiable — runs after every answered query without exception.