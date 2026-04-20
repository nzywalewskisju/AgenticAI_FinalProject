# src/memory/registry.py
# Long-term document registry — persists until the user explicitly clears it.
# Tracks which files have been ingested into the user's ChromaDB collection.
# Each record contains: file_path, file_name, upload_timestamp, chunk_count
# Used by:
#   - GUI document panel: shows user which documents are currently loaded
#   - Ingestion pipeline: skips re-ingestion if file hash matches an existing record
#   - Orchestrator: checks if any documents exist before accepting a query
# Users can remove individual documents or clear all from the GUI.
# Removing a document from the registry also removes its chunks from ChromaDB.
# Persisted to data/registry/{user_id}.json