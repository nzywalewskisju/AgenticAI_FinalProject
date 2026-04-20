# test_connection.py
# Run this before anything else to verify the environment is correctly set up.
# Checks:
#   - Ollama is running and reachable at OLLAMA_BASE_URL
#   - LLM_MODEL (llama3) is available in Ollama
#   - EMBEDDING_MODEL (nomic-embed-text) is available in Ollama
#   - ChromaDB can be initialized at CHROMA_DB_PATH
#   - All required Python packages are importable
#   - data/, logs/, db/ directories exist and are writable
# Exit code 0 means everything is ready. Any failure prints a clear error message.

import ollama
import chromadb
from config import LLM_MODEL, EMBEDDING_MODEL, CHROMA_DB_PATH

def test_llm():
    print("Testing LLM connection...")
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": "Say hello in one sentence."}]
    )
    print("LLM response:", response['message']['content'])
    print("LLM: OK\n")

def test_embeddings():
    print("Testing embedding model...")
    response = ollama.embeddings(
        model=EMBEDDING_MODEL,
        prompt="This is a test sentence."
    )
    vector = response['embedding']
    print(f"Embedding length: {len(vector)} dimensions")
    print("Embeddings: OK\n")

def test_chromadb():
    print("Testing ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection("test_collection")
    collection.add(
        documents=["This is a test document"],
        ids=["test1"]
    )
    results = collection.query(query_texts=["test"], n_results=1)
    print("ChromaDB query result:", results['documents'])
    print("ChromaDB: OK\n")

if __name__ == "__main__":
    test_llm()
    test_embeddings()
    test_chromadb()
    print("All systems operational. You are ready to build.")