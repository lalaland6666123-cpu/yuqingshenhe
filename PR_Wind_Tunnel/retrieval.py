"""案例检索模块 ── BM25 + 轻量 Embedding 混合检索

用法:
    from retrieval import HybridRetriever

    retriever = HybridRetriever()                    # 自动加载 bge-small-zh-v1.5 (~95MB)
    retriever.build(cases)                           # 构建索引
    matches = retriever.search("用户事件描述", top_k=10)  # 混合检索
"""

import json
import os
import re

import numpy as np

# ===== BM25 =====
import jieba
from rank_bm25 import BM25Okapi

# ===== Embedding =====
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 优先用 ModelScope 本地缓存, 其次 HuggingFace
_LOCAL_PATH = os.path.join(BASE_DIR, "models", "BAAI", "bge-small-zh-v1___5")
if os.path.isdir(_LOCAL_PATH):
    MODEL_NAME = _LOCAL_PATH
else:
    MODEL_NAME = "BAAI/bge-small-zh-v1.5"

# RRF 融合参数
RRF_K = 60


class HybridRetriever:
    def __init__(self, model_name=None):
        name = model_name or MODEL_NAME
        is_local = os.path.isdir(name) if name else False
        print(f"加载 Embedding 模型: {name} {'(本地)' if is_local else '(在线)'}")
        self.model = SentenceTransformer(name)
        self.cases = []
        self.case_ids = []
        self.bm25 = None
        self.embeddings = None  # numpy (n_cases, dim)

    def build(self, cases):
        """构建 BM25 和 Embedding 双索引

        Args:
            cases: [{"id":..., "title":..., "crisis_type":..., "description":..., ...}, ...]
        """
        self.cases = cases
        self.case_ids = [c["id"] for c in cases]

        # 拼接每个案例的检索文本
        texts = []
        for c in cases:
            text = (
                f"{c.get('crisis_type', '')}。{c.get('description', '')}。"
                f"{c.get('public_reaction', '')}"
            )
            texts.append(text)

        # ---- BM25 ----
        tokenized = [list(jieba.cut(t)) for t in texts]
        self.bm25 = BM25Okapi(tokenized)

        # ---- Embedding ----
        self.retrieval_texts = texts
        self.embeddings = self.model.encode(
            texts, show_progress_bar=True, normalize_embeddings=True
        )

        print(f"索引构建完成: {len(cases)} 条案例, 向量维度 {self.embeddings.shape[1]}")

    def _bm25_search(self, query, top_k):
        tokens = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokens)
        indices = np.argsort(scores)[::-1][:top_k]
        return list(zip(indices, scores[indices]))

    def _embedding_search(self, query, top_k):
        q_vec = self.model.encode(
            [query], normalize_embeddings=True
        )
        scores = np.dot(self.embeddings, q_vec.T).flatten()
        indices = np.argsort(scores)[::-1][:top_k]
        return list(zip(indices, scores[indices]))

    def search(self, query, top_k=10):
        """混合检索: BM25 + Embedding, RRF 融合

        Returns:
            [{"case": {...}, "bm25_rank": int, "emb_rank": int, "rrf_score": float}, ...]
            按 RRF 分数降序排列
        """
        if not self.cases or not query.strip():
            return []

        # 两路各取 top_k * 2, 给 RRF 更大的候选池
        bm25_hits = self._bm25_search(query, top_k * 2)
        emb_hits = self._embedding_search(query, top_k * 2)

        # RRF 融合
        bm25_rank = {idx: r + 1 for r, (idx, _) in enumerate(bm25_hits)}
        emb_rank = {idx: r + 1 for r, (idx, _) in enumerate(emb_hits)}

        scores = {}
        for idx in bm25_rank:
            b_rank = bm25_rank[idx]
            e_rank = emb_rank.get(idx, len(emb_hits) + 1)
            scores[idx] = 1.0 / (RRF_K + b_rank) + 1.0 / (RRF_K + e_rank)
        for idx in emb_rank:
            if idx not in scores:
                e_rank = emb_rank[idx]
                b_rank = len(bm25_hits) + 1
                scores[idx] = 1.0 / (RRF_K + b_rank) + 1.0 / (RRF_K + e_rank)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for idx, rrf_score in ranked:
            results.append({
                "case": self.cases[idx],
                "bm25_rank": bm25_rank.get(idx, len(bm25_hits) + 1),
                "emb_rank": emb_rank.get(idx, len(emb_hits) + 1),
                "rrf_score": round(rrf_score, 6),
            })
        return results

    def search_bm25_only(self, query, top_k=10):
        """仅 BM25 检索"""
        hits = self._bm25_search(query, top_k)
        return [
            {"case": self.cases[idx], "score": float(score), "rank": i + 1}
            for i, (idx, score) in enumerate(hits)
        ]

    def search_embedding_only(self, query, top_k=10):
        """仅 Embedding 检索"""
        hits = self._embedding_search(query, top_k)
        return [
            {"case": self.cases[idx], "score": float(score), "rank": i + 1}
            for i, (idx, score) in enumerate(hits)
        ]
