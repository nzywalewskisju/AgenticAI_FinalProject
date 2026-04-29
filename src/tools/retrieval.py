# retrieval.py
# Retrieval tools used by the reasoning agent to search policy documents.
# Semantic search uses ChromaDB vector similarity via nomic-embed-text
# embeddings. Keyword search uses BM25 for exact term matching, which
# catches content that semantic search misses due to fragmented PDF tables
# or poor section headers. Reranking uses a secondary LLM call to score
# chunks by true relevance and drop low scorers.
#
# Functions: retrieve_chunks, keyword_search, rerank_results,
#            _get_collection, _embed_query

import requests
from config import (
    CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL,
    OLLAMA_BASE_URL, TOP_K_RESULTS, SIMILARITY_THRESHOLD
)
import chromadb
from rank_bm25 import BM25Okapi
from src.tools.utils import call_llm, safe_json_parse, format_chunks_for_prompt


def _get_collection(user_id: str):
    # Returns the ChromaDB collection scoped to the given user.
    # Collection is named hr_documents_{user_id}.

    client = chromadb.PersistentClient(path=f"{CHROMA_DB_PATH}/{user_id}")
    collection = client.get_or_create_collection(
        name=f"{COLLECTION_NAME}_{user_id}",
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def _embed_query(text: str) -> list[float]:
    # Sends a text string to Ollama's nomic-embed-text model and returns
    # the embedding vector.

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": EMBEDDING_MODEL, "input": text}
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]


def retrieve_chunks(query: str, user_id: str, top_k: int = TOP_K_RESULTS) -> list[dict]:
    # Embeds the query and runs cosine similarity search against the
    # user's ChromaDB collection. Filters results by similarity threshold.

    collection = _get_collection(user_id)

    if collection.count() == 0:
        return []

    embedding = _embed_query(query)

    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(documents, metadatas, distances):
        if dist <= SIMILARITY_THRESHOLD:
            chunks.append({
                "text": doc,
                "metadata": meta,
                "distance": dist
            })

    return chunks


def keyword_search(query: str, user_id: str, top_k: int = TOP_K_RESULTS) -> list[dict]:
    # Pulls all chunks from ChromaDB and ranks them using BM25 exact
    # term matching. Returns top results with score above zero.

    collection = _get_collection(user_id)

    if collection.count() == 0:
        return []

    # Pull all documents for BM25 ranking
    all_results = collection.get(include=["documents", "metadatas"])
    documents = all_results.get("documents", [])
    metadatas = all_results.get("metadatas", [])

    if not documents:
        return []

    # Tokenize for BM25
    tokenized_docs = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Pair documents with scores and sort
    scored = sorted(
        zip(documents, metadatas, scores),
        key=lambda x: x[2],
        reverse=True
    )

    results = []
    for doc, meta, score in scored[:top_k]:
        if score > 0:
            results.append({
                "text": doc,
                "metadata": meta,
                "score": score
            })

    return results


def rerank_results(query: str, chunks: list[dict]) -> list[dict]:
   # Sends all chunks to the LLM in one call and asks it to score each
    # 0 to 10 for relevance. Drops chunks scoring below 4 and sorts the rest.
    
    if not chunks:
        return []

    chunks_text = format_chunks_for_prompt(chunks)

    system_prompt = """You are a relevance scoring assistant.
You will be given a user query and a numbered list of document chunks.
Score each chunk from 0 to 10 based on how relevant and useful it is for answering the query.
10 = directly answers the query with specific policy details
5  = related topic but not directly applicable
0  = irrelevant
Respond only in JSON as a list: [{"index": 1, "score": 8}, {"index": 2, "score": 3}, ...]
Include a score for every chunk. Do not include explanations."""

    prompt = f"Query: {query}\n\nChunks:\n{chunks_text}"

    response = call_llm(prompt, system_prompt=system_prompt)
    scores_raw = safe_json_parse(response, fallback=[])

    # Build score lookup by index (1-based to match format_chunks_for_prompt)
    score_map = {}
    if isinstance(scores_raw, list):
        for item in scores_raw:
            if isinstance(item, dict):
                score_map[item.get("index", -1)] = item.get("score", 0)

    # Attach scores to chunks and filter low scorers
    scored_chunks = []
    for i, chunk in enumerate(chunks, 1):
        score = score_map.get(i, 0)
        if score >= 4:
            scored_chunks.append({**chunk, "rerank_score": score})

    # Sort by rerank score descending
    scored_chunks.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

    return scored_chunks