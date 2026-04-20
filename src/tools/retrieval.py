# src/tools/retrieval.py
# Retrieval tools owned by the Reasoning Sub-Agent.
# Functions:
#   retrieve_chunks(query, user_id, top_k)
#     — dense vector semantic search against the user's ChromaDB collection
#     — filters results by SIMILARITY_THRESHOLD before returning
#   keyword_search(query, user_id, top_k)
#     — sparse BM25 keyword search for exact term matching
#     — complements semantic search for policy-specific terminology
#   rerank_results(query, chunks)
#     — re-scores retrieved chunks for true relevance using a second Llama call
#     — each chunk is scored 0-10, results sorted descending, low scorers dropped
# Never called directly — always called through the ReAct loop in reasoning.py.