"""简单 RAG 引擎验证脚本"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from rag_engine import get_rag_engine, _embedding_fn
from chromadb import PersistentClient

# Step 1: Check collections directly
print("=== Collections check ===")
client = PersistentClient(path="data/chroma_db")
for coll_info in client.list_collections():
    c = client.get_collection(coll_info.name)
    print(f"  {c.name}: {c.count()} items")

# Step 2: Test embedding
print("\n=== Embedding test ===")
emb = _embedding_fn(["测试"])
print(f"  Dim: {len(emb[0])}, Sample: {[round(x,4) for x in emb[0][:5]]}")

# Step 3: Query through RAG engine
print("\n=== RAG Engine query test ===")
rag = get_rag_engine()

# Check if _get_collection works
for lib in ["case_studies", "black_words", "red_lines", "hate_comments"]:
    col = rag._get_collection(lib)
    print(f"  {lib}: {'OK' if col else 'MISSING'} (count={col.count() if col else 0})")

# Query
query = "李佳琦直播眉笔太贵 消费者被羞辱"
print(f"\n  Query: {query}")
result = rag.retrieve(query, top_k=3)
for key, vals in result.items():
    print(f"  {key}: {len(vals)} results")
    for v in vals[:2]:
        print(f"    - sim={v.get('similarity',0):.3f} | {str(list(v.values())[:2])}")

print("\nDone!")

