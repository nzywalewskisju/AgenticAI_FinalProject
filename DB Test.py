import chromadb
client = chromadb.PersistentClient(path="./db/a5152c19-3f3b-45e2-a22e-d6e4f6934312")
collection = client.get_collection("hr_documents_a5152c19-3f3b-45e2-a22e-d6e4f6934312")
results = collection.get(include=["metadatas"])
headers = set(m.get("section_header", "") for m in results["metadatas"])
for h in sorted(headers):
    print(h)