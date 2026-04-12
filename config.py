# config.py

# Model settings
LLM_MODEL = "llama3"
EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"

# Vector DB settings
CHROMA_DB_PATH = "./db"
COLLECTION_NAME = "hr_documents"

# Retrieval settings
TOP_K_RESULTS = 5
SIMILARITY_THRESHOLD = 0.4  # below this score, treat as no result found

# Document ingestion
HR_DOCS_PATH = "./data/hr_docs"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50