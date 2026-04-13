# embedder.py
# responsible for embedding chunks and storing them in chromadb
# uses nomic-embed-text via ollama to convert chunk text into vectors
# stores each chunk with: its vector, its text, and its metadata
# also handles: checking if a document has already been ingested to avoid duplicates
# output: confirmation that chunks were stored successfully
# this file + chunker.py + loader.py form the full ingestion pipeline — run them in order