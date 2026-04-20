# tests/test_tools.py
# Unit tests for all tools in isolation.
# Ollama and ChromaDB are mocked — these tests do not require a running Ollama instance.
# Tests retrieval.py:
#   - retrieve_chunks returns results above SIMILARITY_THRESHOLD only
#   - keyword_search returns results containing exact query terms
#   - rerank_results correctly sorts chunks by Llama-assigned score
# Tests document.py:
#   - check_policy_coverage returns True when relevant content exists
#   - check_policy_coverage returns False when collection is empty or off-topic
#   - registry functions correctly read, write, and delete from JSON
# Tests governance.py:
#   - detect_pii flags queries containing another employee's sensitive information
#   - assess_escalation_risk returns float between 0 and 1
#   - compliance_stamp flags absolute statements correctly
#   - write_audit_log appends valid JSON to audit_log.jsonl
# Tests utils.py:
#   - clean_llm_json_response strips code fences correctly
#   - truncate_text does not exceed max_tokens