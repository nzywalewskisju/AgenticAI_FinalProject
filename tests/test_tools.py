# test_tools.py
# tests for every tool in src/tools/ — test each function in isolation before agents use them
# what to test:
#   - retrieve_chunks returns results for a known query
#   - retrieve_chunks returns empty for a query below the similarity threshold
#   - keyword_search returns results for an exact policy name
#   - check_document_exists returns True for a topic that exists, False for one that does not
#   - detect_pii flags text containing a social security number or full name
#   - get_current_date returns a valid date
# run with: python -m pytest tests/test_tools.py