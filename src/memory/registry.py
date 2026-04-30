# registry.py
# Long-term document registry that persists until the user explicitly clears it.
# Tracks which files have been ingested into the user's ChromaDB collection.
# Re-exports add_to_registry, get_registry, and remove_from_registry from
# src/tools/document.py and adds two registry-level operations: checking
# whether any documents exist and clearing all documents at once.
#
# Functions: has_documents, clear_all_documents,
#            add_to_registry (re-export), get_registry (re-export),
#            remove_from_registry (re-export)

from src.tools.document import (
    add_to_registry,
    get_registry,
    remove_from_registry
)


def has_documents(user_id: str) -> bool:
    # Returns True if the user has at least one ingested document.
    # Used by the orchestrator to block queries when no documents are loaded.

    registry = get_registry(user_id)
    return len(registry) > 0


def clear_all_documents(user_id: str) -> int:
    # Deletes the entire ChromaDB collection and clears the registry file
    # for the user. Returns the number of documents removed.

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