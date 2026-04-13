# governance.py — TOOLS layer (not the agent)
# low-level functions that the Governance Sub-Agent calls
# functions defined here:
#   - detect_pii(text): scans input for names, SSNs, or other employee private info
#   - assess_escalation_risk(query): scores whether a query needs a human HR rep
#   - write_audit_log(query, chunks_used, answer, agent_trace): logs full interaction to file
#   - compliance_stamp(answer): checks final answer for legally dangerous language
# note: this is different from src/agents/governance.py which is the reasoning agent
#       this file is just the raw functions — the agent is what decides when to call them