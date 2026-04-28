import chromadb

USER_ID = "27ee5d21-fd56-4372-8c47-d01c095f9e9f"

client = chromadb.PersistentClient(path=f"./db/{USER_ID}")
collection = client.get_collection(f"hr_documents_{USER_ID}")
results = collection.get(include=["documents", "metadatas"])

for doc, meta in zip(results["documents"], results["metadatas"]):
    if meta.get("chunk_index") in [35, 36, 37, 38, 39]:
        print(f"\nChunk index: {meta.get('chunk_index')}")
        print(f"Section: {meta.get('section_header')}")
        print(doc)
        print("---")
