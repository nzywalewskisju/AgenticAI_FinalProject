# document.py
# contains tools related to checking what exists in the vector database
# tools defined here:
#   - check_document_exists(topic): checks if any chunks related to a topic exist in the DB
#     returns True/False — if False, the Reasoning Agent should not attempt retrieval
#     this is the first line of defense against hallucination on nonexistent policies
# called by the Reasoning Sub-Agent before retrieve_chunks