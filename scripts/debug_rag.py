"""Debug RAG engine"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from chromadb import PersistentClient

client = PersistentClient(path="data/chroma_db")
collections = client.list_collections()
print(f"Total collections: {len(collections)}")
for c in collections:
    print(f"  - {c.name}: {c.count()} items")

# Test embedding function
from rag_engine import _embedding_fn
test_doc = ["测试文本"]
try:
    emb = _embedding_fn(test_doc)
    print(f"\nEmbedding dim: {len(emb[0])}")
    print(f"Embedding sample: {emb[0][:5]}...")
except Exception as e:
    print(f"\nEmbedding error: {e}")

# Try query
if collections:
    col = client.get_collection("case_studies")
    results = col.query(query_texts=["李佳琦直播眉笔太贵"], n_results=3)
    print(f"\nQuery results: {len(results['ids'][0]) if results['ids'] else 0}")
    if results['ids'] and results['ids'][0]:
        for i, rid in enumerate(results['ids'][0]):
            dist = results['distances'][0][i]
            meta = results['metadatas'][0][i]
            print(f"  {rid} (dist={dist:.3f}): {meta.get('title','')}")
