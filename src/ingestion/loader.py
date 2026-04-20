# src/ingestion/loader.py
# Responsible for reading raw files into text and metadata.
# Accepts a list of file paths — files can come from anywhere on the user's disk.
# Supports PDF (via LangChain PyPDFLoader) and DOCX (via python-docx).
# Preserves document structure by detecting and marking headings with ## prefix
# so the chunker can use them as section boundaries.
# Returns a list of dicts: {text, metadata} where metadata contains:
#   source_file, file_type, document_name, user_id
# Never called directly by agents — called by the ingestion pipeline only.