# src/ingestion/chunker.py
# Responsible for splitting loaded documents into chunks for embedding.
# Uses a hybrid chunking strategy in this order:
#   1. Heading/section detection — splits on ## markers and numbered section headers
#   2. Paragraph fallback — splits on double newlines if no headings found
#   3. Fixed-size fallback — uses RecursiveCharacterTextSplitter if paragraphs are too large
# Each chunk carries metadata: source_file, file_type, document_name,
#   section_header, chunk_index, user_id
# Chunk size and overlap are set in config.py (CHUNK_SIZE, CHUNK_OVERLAP).
# Never called directly by agents — called by the ingestion pipeline only.