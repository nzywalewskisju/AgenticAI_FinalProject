# test_agents.py
# tests for the agent layer — run these after tools are confirmed working
# what to test:
#   - orchestrator rejects an out-of-scope query without calling any sub-agents
#   - orchestrator routes a valid HR query to the reasoning agent
#   - reasoning agent returns a draft answer with source chunks attached
#   - reasoning agent returns "no information found" when no relevant chunks exist
#   - review agent rejects a draft answer that contains an ungrounded claim
#   - governance agent escalates a high-risk query without returning an answer
# run with: python -m pytest tests/test_agents.py