# src/tools/document.py
# Document and registry tools owned by the Reasoning Sub-Agent.
# Functions:
#   check_policy_coverage(topic, user_id)
#     — checks whether the user's ChromaDB collection contains content relevant
#       to the topic before attempting retrieval
#     — MUST be called before retrieve_chunks in the ReAct loop
#     — prevents retrieval attempts when no relevant policy exists
#   list_available_topics(user_id)
#     — returns a summary of what policy topics are covered in the user's documents
#     — used to inform the user what the system can and cannot answer
# Document registry functions:
#   add_to_registry(user_id, file_path, chunk_count)
#     — records a successfully ingested document
#   get_registry(user_id)
#     — returns the full document registry for a user
#   remove_from_registry(user_id, file_path)
#     — removes a document record when the user clears it
#   registry is persisted to data/registry/{user_id}.json

import json
import os
import hashlib
from datetime import datetime
from config import CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL, OLLAMA_BASE_URL
import chromadb
import requests


def _get_collection(user_id: str):
    """
    Returns the ChromaDB collection scoped to this user.
    Collection name: hr_documents_{user_id}
    """
    client = chromadb.PersistentClient(path=f"{CHROMA_DB_PATH}/{user_id}")
    collection = client.get_or_create_collection(
        name=f"{COLLECTION_NAME}_{user_id}",
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def _get_registry_path(user_id: str) -> str:
    return f"./data/registry/{user_id}.json"


def _load_registry(user_id: str) -> list:
    path = _get_registry_path(user_id)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_registry(user_id: str, registry: list) -> None:
    path = _get_registry_path(user_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def _embed_query(text: str) -> list[float]:
    """
    Embeds a query string using Ollama's nomic-embed-text model.
    Must match the embedding model used at ingestion time.
    """
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": EMBEDDING_MODEL, "prompt": text}
    )
    response.raise_for_status()
    return response.json()["embedding"]


def check_policy_coverage(topic: str, user_id: str) -> dict:
    """
    Checks whether the user's ChromaDB collection contains content
    relevant to the topic before attempting retrieval.
    MUST be called before retrieve_chunks in the ReAct loop.
    Returns {covered: bool, reason: str}
    """
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
            n_results=min(3, collection.count()),
            include=["distances", "documents"]
        )

        distances = results.get("distances", [[]])[0]
        if not distances:
            return {"covered": False, "reason": "No results returned from collection."}

        best_distance = min(distances)
        # Cosine distance: lower = more similar. 0.5 is a loose coverage threshold.
        covered = best_distance < 0.5

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
    """
    Returns a summary of what policy topics are covered in the user's documents.
    Used to inform the user what the system can and cannot answer.
    Returns {topics: list[str], document_count: int}
    """
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
    """
    Records a successfully ingested document in the user's registry.
    Uses a hash of the file path to detect re-uploads of the same file.
    Returns the new registry record.
    """
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
    """
    Returns the full document registry for a user.
    Used by the GUI document panel and the orchestrator.
    """
    return _load_registry(user_id)


def remove_from_registry(user_id: str, file_path: str) -> bool:
    """
    Removes a document record from the registry and deletes its chunks
    from the user's ChromaDB collection.
    Returns True if removed, False if not found.
    """
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