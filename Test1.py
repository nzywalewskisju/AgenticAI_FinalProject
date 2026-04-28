import chromadb

USER_ID = "c2fecc92-8195-47a6-9913-78824671c53e"

client = chromadb.PersistentClient(path=f"./db/{USER_ID}")
collection = client.get_collection(f"hr_documents_{USER_ID}")
results = collection.get(include=["documents", "metadatas"])

for doc, meta in zip(results["documents"], results["metadatas"]):
    if "voluntary" in meta.get("section_header", "").lower() or "nationwide" in doc.lower():
        print(f"Section: {meta.get('section_header')}")
        print(doc[:500])
        print("---")