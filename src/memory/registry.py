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

# NOTE: Registry logic is implemented in src/tools/document.py
# (add_to_registry, get_registry, remove_from_registry)
# This module re-exports those functions for clean imports from the memory package.

from src.tools.document import (
    add_to_registry,
    get_registry,
    remove_from_registry
)


def has_documents(user_id: str) -> bool:
    """
    Returns True if the user has at least one ingested document.
    Used by the orchestrator to gate queries when no documents are loaded.
    """
    registry = get_registry(user_id)
    return len(registry) > 0


def clear_all_documents(user_id: str) -> int:
    """
    Removes all documents from the user's registry and ChromaDB collection.
    Returns the number of documents removed.
    Called when user clicks 'Clear All Documents' in the GUI.
    """
    import chromadb
    from config import CHROMA_DB_PATH, COLLECTION_NAME

    registry = get_registry(user_id)
    count = len(registry)

    # Wipe the ChromaDB collection entirely
    try:
        client = chromadb.PersistentClient(path=f"{CHROMA_DB_PATH}/{user_id}")
        client.delete_collection(name=f"{COLLECTION_NAME}_{user_id}")
    except Exception as e:
        print(f"[REGISTRY] Warning: could not clear ChromaDB collection: {e}")

    # Wipe the registry file
    import json
    import os
    path = f"./data/registry/{user_id}.json"
    if os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump([], f)

    return count


__all__ = [
    "add_to_registry",
    "get_registry",
    "remove_from_registry",
    "has_documents",
    "clear_all_documents"
]