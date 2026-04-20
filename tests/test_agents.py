# tests/test_agents.py
# Integration tests for agent behaviour.
# Ollama is mocked with controlled responses to test routing and logic.
# Tests orchestrator.py:
#   - out_of_scope queries are rejected before any agent is called
#   - high_stakes queries are escalated before Reasoning runs
#   - queries with no ingested documents prompt a file upload request
#   - profile facts are extracted and persisted correctly
#   - no-chunks guard rejects answers produced without retrieved evidence
#   - max 2 retries are enforced when Review rejects Reasoning output
# Tests reasoning.py:
#   - ReAct loop calls check_policy_coverage before retrieve_chunks
#   - loop exits correctly when Answer: is detected in response
#   - loop exits after MAX_REACT_TURNS if no answer is produced
#   - returns situation_facts, relevant_policy, advice, chunks_used in output
# Tests review.py:
#   - all five checks run in order
#   - answer is rejected if grounding score < 0.7
#   - answer is rejected if advice does not apply policy to user situation
# Tests governor.py:
#   - PII queries are blocked at pre-check
#   - escalation threshold correctly routes high-risk queries
#   - audit log entry contains all required fields