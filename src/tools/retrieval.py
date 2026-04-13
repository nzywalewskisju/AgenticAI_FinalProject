# retrieval.py
# contains all tools related to searching the vector database
# tools defined here:
#   - retrieve_chunks(query, filters): dense vector similarity search, returns top-k chunks
#   - keyword_search(term): sparse BM25 keyword search for exact policy names or numbers
#   - rerank_results(query, chunks): re-scores retrieved chunks for true relevance before use
# these are called by the Reasoning Sub-Agent during the ReAct loop
# similarity scores below SIMILARITY_THRESHOLD (set in config.py) should return empty