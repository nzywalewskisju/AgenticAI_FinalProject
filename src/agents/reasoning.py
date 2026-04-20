# src/agents/reasoning.py
# Reasoning Sub-Agent — implements the ReAct loop.
# This is what makes the system an agent rather than a retrieval chatbot.
# The agent dynamically decides which tools to call, in what order, and how many times.
# ReAct loop structure: Thought → Action → PAUSE → Observation → repeat
# Max iterations: MAX_REACT_TURNS (set in config.py)
# Tools available to this agent:
#   - check_policy_coverage (MUST be called before retrieve_chunks)
#   - retrieve_chunks
#   - keyword_search
#   - rerank_results
#   - get_current_date
#   - request_clarification
# The agent explicitly:
#   1. Extracts the facts of the user's situation from the query
#   2. Retrieves policy relevant to those facts
#   3. Reasons about the gap between the user's situation and policy requirements
#   4. Produces concrete, personalized advice — not just a policy summary
# Returns: {situation_facts, relevant_policy, advice, chunks_used, status, iterations}
# Status: "success" | "clarification" | "no_info" | "error"