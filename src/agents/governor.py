# governor.py — AGENT layer
# the Governance Sub-Agent — called twice per query: before and after reasoning
# pre-check (called by orchestrator before reasoning):
#   - runs detect_pii on the incoming query
#   - runs assess_escalation_risk to decide if a human HR rep should handle this instead
#   - blocks the pipeline and escalates if risk is too high
# post-check (called after review sub-agent approves the answer):
#   - runs compliance_stamp on the final answer
#   - runs write_audit_log to record the full interaction
# note: this is the agent that decides when to call the tools in src/tools/governance.py