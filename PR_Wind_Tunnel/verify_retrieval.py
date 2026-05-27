"""验证检索方案: 用现有案例自测 Top-N 命中率

测试逻辑: 拿每条案例的 description 作为查询, 检查检索结果 Top-N 是否包含自身。
这是 leave-one-out 自检 —— 最严格的测试, 实际使用中用户输入不会完全和案例描述一模一样。
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retrieval import HybridRetriever

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, "case_studies.json")

with open(JSON_PATH, "r", encoding="utf-8") as f:
    CASES = json.load(f)

print(f"加载 {len(CASES)} 条案例")

retriever = HybridRetriever()
retriever.build(CASES)

# ---- 自检: 每条案例的 description 查自己 ----
print("\n" + "=" * 60)
print("自检: 用案例自身的 description 检索, 看自身是否在 Top-N 内")
print("=" * 60)

for top_n in [1, 3, 5, 10]:
    hits = 0
    for case in CASES:
        query = case["description"]
        results = retriever.search(query, top_k=top_n)
        result_ids = [r["case"]["id"] for r in results]
        if case["id"] in result_ids:
            hits += 1

    pct = hits / len(CASES) * 100
    print(f"  Top-{top_n:2d}: {hits}/{len(CASES)} ({pct:.1f}%)")

# ---- 分别看 BM25 和 Embedding 单独表现 ----
print("\n" + "=" * 60)
print("对比: BM25 单独 vs Embedding 单独 vs 混合")
print("=" * 60)

for top_n in [5, 10]:
    bm25_hits = emb_hits = hybrid_hits = 0
    for case in CASES:
        query = case["description"]

        hy = retriever.search(query, top_k=top_n)
        if case["id"] in [r["case"]["id"] for r in hy]:
            hybrid_hits += 1

        bm = retriever.search_bm25_only(query, top_k=top_n)
        if case["id"] in [r["case"]["id"] for r in bm]:
            bm25_hits += 1

        em = retriever.search_embedding_only(query, top_k=top_n)
        if case["id"] in [r["case"]["id"] for r in em]:
            emb_hits += 1

    n = len(CASES)
    print(f"  Top-{top_n:2d}: BM25={bm25_hits}/{n} ({bm25_hits/n*100:.1f}%)  "
          f"Embedding={emb_hits}/{n} ({emb_hits/n*100:.1f}%)  "
          f"混合={hybrid_hits}/{n} ({hybrid_hits/n*100:.1f}%)")

# ---- 展示几个样例 ----
print("\n" + "=" * 60)
print("样例展示: 随机几条案例的检索结果")
print("=" * 60)

samples = [CASES[0], CASES[15], CASES[30], CASES[50], CASES[80]]
for case in samples:
    print(f"\n查询案例: [{case['id']}] {case['title']}")
    print(f"  crisis_type: {case['crisis_type']}")
    results = retriever.search(case["description"], top_k=5)
    for i, r in enumerate(results):
        hit = "[HIT]" if r["case"]["id"] == case["id"] else "     "
        print(f"  {hit} #{i+1} [{r['case']['id']}] {r['case']['title']} "
              f"(BM25排{r['bm25_rank']} / Emb排{r['emb_rank']})")

# ---- 交叉扰动测试: 改写 query 看鲁棒性 ----
print("\n" + "=" * 60)
print("交叉测试: 用其他案例的 description 查询, 看返回是否合理")
print("=" * 60)

# 取几条案例作为 query, 看能否匹配到同类案例
test_queries = [
    ("某餐饮品牌被曝光厨房卫生问题", "预期匹配: 食品安全类"),
    ("品牌创始人在公开场合发表了歧视性言论, 引发全网愤怒", "预期匹配: 傲慢发言类"),
    ("因数据泄露导致千万用户信息被卖到暗网", "预期匹配: 隐私/网络安全类"),
    ("新能源车发生自燃事故, 车主维权", "预期匹配: 汽车安全类"),
]
for query, expected in test_queries:
    print(f"\n查询: {query}")
    print(f"  {expected}")
    results = retriever.search(query, top_k=3)
    for i, r in enumerate(results):
        print(f"  #{i+1} [{r['case']['id']}] {r['case']['title']} "
              f"({r['case']['crisis_type']})")

print("\n验证完成!")
