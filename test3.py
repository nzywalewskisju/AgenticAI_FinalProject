import chromadb

USER_ID = "a6218b2a-f59d-4f99-9466-1d455b46b0be"

client = chromadb.PersistentClient(path=f"./db/{USER_ID}")
collection = client.get_collection(f"hr_documents_{USER_ID}")
results = collection.get(include=["documents", "metadatas"])

for doc, meta in zip(results["documents"], results["metadatas"]):
    if "secure" in doc.lower() or "11,250" in doc or "34,750" in doc or "catch-up" in doc.lower():
        print(f"\nSection: {meta.get('section_header')}")
        print(f"Chunk index: {meta.get('chunk_index')}")
        print(doc[:500])
        print("---")