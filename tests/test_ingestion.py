# tests/test_ingestion.py
# Unit tests for the ingestion pipeline.
# Tests loader.py:
#   - PDF loading returns expected text and metadata structure
#   - DOCX loading returns expected text and metadata structure
#   - Heading detection correctly marks ## prefixes
#   - Unsupported file types raise a clear error
# Tests chunker.py:
#   - Heading-based chunking splits on correct boundaries
#   - Paragraph fallback activates when no headings are detected
#   - Fixed-size fallback activates when paragraphs exceed CHUNK_SIZE
#   - All chunks carry required metadata fields
# Tests embedder.py:
#   - Chunks are stored with correct metadata in ChromaDB
#   - Duplicate ingestion of the same file does not create duplicate chunks
#   - Collection is correctly scoped by user_id