# document.py
# Document coverage checking and registry management.
# check_policy_coverage runs a lightweight semantic search to verify
# relevant policy exists before full retrieval is attempted.
# list_available_topics summarizes what policy areas are covered in
# the user's ingested documents. Registry functions track ingested
# documents per user so the GUI can display and manage them.
#
# Functions: check_policy_coverage, list_available_topics,
#            add_to_registry, get_registry, remove_from_registry,
#            _get_collection, _embed_query, _get_registry_path,
#            _load_registry, _save_registry

import json
import os
import hashlib
from datetime import datetime
from config import CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL, OLLAMA_BASE_URL
import chromadb
import requests


def _get_collection(user_id: str):
    # Returns the ChromaDB collection scoped to the given user.

    client = chromadb.PersistentClient(path=f"{CHROMA_DB_PATH}/{user_id}")
    collection = client.get_or_create_collection(
        name=f"{COLLECTION_NAME}_{user_id}",
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def _get_registry_path(user_id: str) -> str:
     # Returns the file path for the user's registry JSON file.
    return f"./data/registry/{user_id}.json"


def _load_registry(user_id: str) -> list:
    # Reads and returns the user's registry JSON from disk.
    # Returns an empty list if the file does not exist yet.

    path = _get_registry_path(user_id)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_registry(user_id: str, registry: list) -> None:
    # Writes the registry list to the user's registry JSON file on disk.

    path = _get_registry_path(user_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def _embed_query(text: str) -> list[float]:
    # Embeds a query string using nomic-embed-text via Ollama.

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": EMBEDDING_MODEL, "input": text}
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]


def check_policy_coverage(topic: str, user_id: str) -> dict:
    # Runs a lightweight semantic search to verify relevant policy exists
    # before full retrieval is attempted. Must be called before retrieve_chunks.

    try:
        collection = _get_collection(user_id)
        if collection.count() == 0:
            return {
                "covered": False,
                "reason": "No documents have been ingested yet for this user."
            }

        embedding = _embed_query(topic)
        results = collection.query(
            query_embeddings=[embedding],
            n_results=min(5, collection.count()),  # was 3
            include=["distances", "documents"]
        )

        distances = results.get("distances", [[]])[0]
        if not distances:
            return {"covered": False, "reason": "No results returned from collection."}

        best_distance = min(distances)
        # Cosine distance: lower = more similar. 0.5 is a loose coverage threshold.
        covered = best_distance < 0.7  # was 0.5

        return {
            "covered": covered,
            "reason": (
                f"Best match distance: {best_distance:.3f}. "
                f"{'Relevant policy content found.' if covered else 'No sufficiently relevant content found.'}"
            )
        }

    except Exception as e:
        return {"covered": False, "reason": f"Coverage check failed: {str(e)}"}


def list_available_topics(user_id: str) -> dict:
    # Returns a list of policy topic names based on the filenames of
    # documents the user has ingested.

    registry = _load_registry(user_id)
    if not registry:
        return {"topics": [], "document_count": 0}

    topics = []
    for record in registry:
        name = record.get("file_name", "")
        if name:
            # Use filename as a proxy for topic — e.g. "pto_policy.pdf" → "PTO Policy"
            topic = os.path.splitext(name)[0].replace("_", " ").replace("-", " ").title()
            topics.append(topic)

    return {
        "topics": topics,
        "document_count": len(registry)
    }


def add_to_registry(user_id: str, file_path: str, chunk_count: int) -> dict:
    # Records a successfully ingested document in the user's registry file.
    # Skips silently if the document is already registered.

    registry = _load_registry(user_id)

    file_hash = hashlib.md5(file_path.encode()).hexdigest()

    # Check for existing entry — skip if already registered
    for record in registry:
        if record.get("file_hash") == file_hash:
            return record

    record = {
        "file_path": file_path,
        "file_name": os.path.basename(file_path),
        "file_hash": file_hash,
        "upload_timestamp": datetime.utcnow().isoformat(),
        "chunk_count": chunk_count
    }

    registry.append(record)
    _save_registry(user_id, registry)
    return record


def get_registry(user_id: str) -> list:
    # Returns the full list of ingested document records for a user.
    # Used by the GUI document panel and the orchestrator availability check.

    return _load_registry(user_id)


def remove_from_registry(user_id: str, file_path: str) -> bool:
    # Removes a document from the registry and deletes its chunks from
    # ChromaDB. Returns True if removed, False if not found.
    
    registry = _load_registry(user_id)
    file_hash = hashlib.md5(file_path.encode()).hexdigest()

    original_count = len(registry)
    registry = [r for r in registry if r.get("file_hash") != file_hash]

    if len(registry) == original_count:
        return False

    # Remove chunks from ChromaDB that belong to this file
    try:
        collection = _get_collection(user_id)
        file_name = os.path.basename(file_path)
        results = collection.get(where={"source_file": file_name})
        ids_to_delete = results.get("ids", [])
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
    except Exception as e:
        print(f"[REGISTRY] Warning: could not remove chunks from ChromaDB: {e}")

    _save_registry(user_id, registry)
    return True