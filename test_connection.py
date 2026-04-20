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

import sys
import os


def check_packages():
    print("Checking required packages...")
    required = [
        "ollama", "chromadb", "bcrypt", "requests",
        "pydantic", "dotenv", "langchain", "langchain_community",
        "langchain_text_splitters", "rank_bm25", "docx", "pypdf"
    ]
    missing = []
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    if missing:
        print(f"  ✗ Missing packages: {', '.join(missing)}")
        print("  Run: pip install -r requirements.txt")
        return False
    print("  ✓ All packages importable")
    return True


def check_directories():
    print("Checking required directories...")
    required_dirs = [
        "./db", "./logs", "./data/profiles",
        "./data/registry", "./data/users"
    ]
    all_ok = True
    for d in required_dirs:
        os.makedirs(d, exist_ok=True)
        if os.access(d, os.W_OK):
            print(f"  ✓ {d}")
        else:
            print(f"  ✗ {d} — not writable")
            all_ok = False
    return all_ok


def check_llm():
    print("Checking LLM connection...")
    import ollama
    from config import LLM_MODEL
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "Say hello in one sentence."}]
        )
        print(f"  ✓ LLM response: {response['message']['content'][:80]}")
        return True
    except Exception as e:
        print(f"  ✗ LLM failed: {e}")
        print(f"  Make sure Ollama is running and '{LLM_MODEL}' is pulled.")
        print(f"  Run: ollama pull {LLM_MODEL}")
        return False


def check_embeddings():
    print("Checking embedding model...")
    import ollama
    from config import EMBEDDING_MODEL
    try:
        response = ollama.embed(
            model=EMBEDDING_MODEL,
            input="This is a test sentence."
        )
        vector = response['embeddings'][0]
        print(f"  ✓ Embedding dimensions: {len(vector)}")
        return True
    except Exception as e:
        print(f"  ✗ Embeddings failed: {e}")
        print(f"  Make sure '{EMBEDDING_MODEL}' is pulled.")
        print(f"  Run: ollama pull {EMBEDDING_MODEL}")
        return False


def check_chromadb():
    print("Checking ChromaDB...")
    import chromadb
    from config import CHROMA_DB_PATH
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_or_create_collection("_connection_test")
        collection.add(
            documents=["This is a test document."],
            ids=["_test_1"]
        )
        results = collection.query(query_texts=["test"], n_results=1)
        assert results["documents"][0][0] == "This is a test document."
        # Clean up — delete the test collection so it doesn't persist
        client.delete_collection("_connection_test")
        print(f"  ✓ ChromaDB read/write at {CHROMA_DB_PATH}")
        return True
    except Exception as e:
        print(f"  ✗ ChromaDB failed: {e}")
        return False


def main():
    print("\n=== HR Policy Assistant — Connection Test ===\n")

    results = {
        "Packages":    check_packages(),
        "Directories": check_directories(),
        "LLM":         check_llm(),
        "Embeddings":  check_embeddings(),
        "ChromaDB":    check_chromadb(),
    }

    print("\n=== Summary ===")
    all_passed = True
    for name, passed in results.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n✓ All systems operational. You are ready to run the assistant.\n")
        sys.exit(0)
    else:
        print("\n✗ Some checks failed. Fix the issues above before running the assistant.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()