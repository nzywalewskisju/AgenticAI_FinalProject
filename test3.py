import chromadb

USER_ID = "27ee5d21-fd56-4372-8c47-d01c095f9e9f"

client = chromadb.PersistentClient(path=f"./db/{USER_ID}")
collection = client.get_collection(f"hr_documents_{USER_ID}")
results = collection.get(include=["documents", "metadatas"])

for doc, meta in zip(results["documents"], results["metadatas"]):
    if meta.get("chunk_index") in [21, 22, 23]:
        if "nexarion remote" in meta.get("document_name", "").lower() or "supplement" in meta.get("source_file", "").lower():
            print(f"\nSection: {meta.get('section_header')}")
            print(f"Chunk index: {meta.get('chunk_index')}")
            print(doc)
            print("---")