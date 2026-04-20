# src/agents/orchestrator.py
# Top-level orchestrator — main entry point for all queries.
# Responsibilities:
#   - classify_query: route to hr_in_scope, high_stakes, or out_of_scope
#   - manage_session_memory: attach prior conversation turns to the query context
#   - extract_profile_facts: detect personal facts in the query and persist to profile
#   - check_document_availability: if no documents ingested for this user, ask for a file
#   - coordinate the full agent pipeline in order:
#       Governor pre-check → Reasoning → Review → Governor post-check
#   - enforce no-chunks guard: reject any answer produced without retrieved evidence
#   - enforce max 2 retries if Review rejects the Reasoning output
#   - return final answer or escalation/clarification message to the GUI
# Uses RoutingDecision (Pydantic) for structured classification output.