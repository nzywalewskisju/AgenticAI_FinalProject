# test_ingestion.py
# tests for the ingestion pipeline — run these after building src/ingestion/
# what to test:
#   - loader.py correctly reads a sample PDF and returns text
#   - loader.py correctly reads a sample .docx and returns text
#   - chunker.py splits text into chunks with correct metadata attached
#   - embedder.py stores chunks in chromadb and they can be retrieved
# run with: python -m pytest tests/test_ingestion.py