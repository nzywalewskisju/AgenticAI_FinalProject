# src/ingestion/embedder.py
# Responsible for embedding chunks and storing them in ChromaDB.
# Uses Ollama's nomic-embed-text model for all embeddings — runs locally, no external calls.
# CRITICAL: The embedding model used here must match the one used at retrieval time.
#   Both use nomic-embed-text. Never change one without changing the other.
# Each user gets their own ChromaDB collection: hr_documents_{user_id}
# Uses deterministic chunk IDs (source_file + chunk_index) to prevent
#   duplicate ingestion if the same file is uploaded again.
# Collection uses cosine similarity space (hnsw:space: cosine).

import requests
import chromadb
from config import (
    CHROMA_DB_PATH, COLLECTION_NAME,
    EMBEDDING_MODEL, OLLAMA_BASE_URL
)

BATCH_SIZE = 32


def _get_collection(user_id: str):
    """
    Returns the ChromaDB collection scoped to this user.
    Creates it if it does not exist.
    """
    client = chromadb.PersistentClient(path=f"{CHROMA_DB_PATH}/{user_id}")
    collection = client.get_or_create_collection(
        name=f"{COLLECTION_NAME}_{user_id}",
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def _embed_text(text: str) -> list[float]:
    """
    Embeds a single text string using Ollama's nomic-embed-text model.
    Used at retrieval time for query embedding — kept for compatibility
    with retrieval.py and document.py which embed single queries.
    """
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": EMBEDDING_MODEL, "prompt": text}
    )
    response.raise_for_status()
    return response.json()["embedding"]


def _embed_texts_batch(texts: list[str]) -> list[list[float]]:
    """
    Embeds multiple texts in a single Ollama API call.
    Much faster than one call per chunk at ingestion time.
    Uses the /api/embed endpoint which accepts a list of inputs.
    """
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": EMBEDDING_MODEL, "input": texts}
    )
    response.raise_for_status()
    return response.json()["embeddings"]


def _make_chunk_id(metadata: dict) -> str:
    """
    Creates a deterministic chunk ID from source_file and chunk_index.
    Prevents duplicate ingestion if the same file is uploaded again.
    """
    source = metadata.get("source_file", "unknown")
    index = metadata.get("chunk_index", 0)
    return f"{source}__chunk_{index}"


def embed_and_store(chunks: list[dict], user_id: str) -> int:
    """
    Embeds all chunks and stores them in the user's ChromaDB collection.
    Processes chunks in batches of BATCH_SIZE for significantly faster ingestion.
    Skips chunks whose ID already exists in the collection.
    Returns the number of new chunks stored.
    """
    collection = _get_collection(user_id)
    stored_count = 0

    # Filter out duplicates first in one pass before any embedding occurs
    new_chunks = []
    for chunk in chunks:
        chunk_id = _make_chunk_id(chunk["metadata"])
        existing = collection.get(ids=[chunk_id])
        if existing and existing.get("ids"):
            print(f"[EMBEDDER] Skipping duplicate chunk: {chunk_id}")
            continue
        new_chunks.append(chunk)

    if not new_chunks:
        print(f"[EMBEDDER] All chunks already ingested. Nothing new to store.")
        return 0

    print(f"[EMBEDDER] Embedding {len(new_chunks)} new chunks in batches of {BATCH_SIZE}...")

    # Process in batches
    for i in range(0, len(new_chunks), BATCH_SIZE):
        batch = new_chunks[i:i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        ids = [_make_chunk_id(c["metadata"]) for c in batch]
        metadatas = [c["metadata"] for c in batch]

        try:
            embeddings = _embed_texts_batch(texts)
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            stored_count += len(batch)
            print(f"[EMBEDDER] {stored_count}/{len(new_chunks)} chunks stored...", end="\r")
        except Exception as e:
            print(f"[EMBEDDER] Warning: batch failed at index {i}: {e}")

    print(f"\n[EMBEDDER] Stored {stored_count} new chunks for user {user_id}")
    return stored_count


def run_ingestion_pipeline(
    file_paths: list[str],
    user_id: str
) -> dict:
    """
    Full ingestion pipeline — loads, chunks, embeds, and stores documents.
    Called inline when a user uploads new files.
    Returns {files_processed, chunks_stored, skipped_files}
    """
    from src.ingestion.loader import load_all_documents
    from src.ingestion.chunker import chunk_all_documents
    from src.tools.document import add_to_registry
    import os

    print(f"[INGESTION] Starting pipeline for {len(file_paths)} file(s)...")

    documents = load_all_documents(file_paths, user_id)
    if not documents:
        return {"files_processed": 0, "chunks_stored": 0, "skipped_files": file_paths}

    chunks = chunk_all_documents(documents)
    chunks_stored = embed_and_store(chunks, user_id)

    # Register each successfully processed file
    processed_files = list({c["metadata"]["source_file"] for c in chunks})
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        if file_name in processed_files:
            file_chunks = [c for c in chunks if c["metadata"]["source_file"] == file_name]
            add_to_registry(user_id, file_path, len(file_chunks))

    print(f"[INGESTION] Complete. {chunks_stored} chunks stored.")
    return {
        "files_processed": len(documents),
        "chunks_stored": chunks_stored,
        "skipped_files": []
    }