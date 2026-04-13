# chunker.py
# responsible for splitting raw document text into chunks that get stored in the vector DB
# do NOT use fixed token size chunking — chunk by document structure instead:
#   - split on section headers, numbered clauses, or double newlines
#   - each chunk should represent one complete idea or policy section
# each chunk must include metadata: source filename, document type, section header, effective date
# output: list of dicts, each containing chunk text and its metadata
