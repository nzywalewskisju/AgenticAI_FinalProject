# src/tools/document.py
# Document and registry tools owned by the Reasoning Sub-Agent.
# Functions:
#   check_policy_coverage(topic, user_id)
#     — checks whether the user's ChromaDB collection contains content relevant
#       to the topic before attempting retrieval
#     — MUST be called before retrieve_chunks in the ReAct loop
#     — prevents retrieval attempts when no relevant policy exists
#   list_available_topics(user_id)
#     — returns a summary of what policy topics are covered in the user's documents
#     — used to inform the user what the system can and cannot answer
# Document registry functions:
#   add_to_registry(user_id, file_path, chunk_count)
#     — records a successfully ingested document
#   get_registry(user_id)
#     — returns the full document registry for a user
#   remove_from_registry(user_id, file_path)
#     — removes a document record when the user clears it
#   registry is persisted to data/registry/{user_id}.json