# src/ingestion/embedder.py
# Responsible for embedding chunks and storing them in ChromaDB.
# Uses Ollama's nomic-embed-text model for all embeddings — runs locally, no external calls.
# CRITICAL: The embedding model used here must match the one used at retrieval time.
#   Both use nomic-embed-text. Never change one without changing the other.
# Each user gets their own ChromaDB collection: hr_documents_{user_id}
# Uses deterministic chunk IDs (source_file + chunk_index) to prevent
#   duplicate ingestion if the same file is uploaded again.
# Collection uses cosine similarity space (hnsw:space: cosine).