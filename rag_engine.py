"""
RAG 检索引擎 v2.0 — 分类框架驱动 + 关键词匹配的混合检索

核心理念变更（v2.0 vs v1.0）:
- v1.0: 语义检索相似案例 → 模仿历史
- v2.0: 关键词 + 语义匹配危机模式 → 引导思考角度

四个物理存储库（ChromaDB 向量索引）:
- red_lines:      绝对红线库
- black_words:    黑词与烂梗库
- hate_comments:  网民恶评库
- case_studies:   历史案例库（辅助参考）

三个逻辑知识库（JSON 直接加载，不建向量索引——关键词匹配为主）:
- crisis_taxonomy.json:     危机模式分类框架（8大类40+子类）
- keyword_templates.json:   关键词模板库（角色×场景×触发词）
- audit_checklist.json:     智能体审查清单（4阶段TODO列表）
"""

import json
import os
from typing import Sequence

from chromadb import PersistentClient
from chromadb.api.types import Documents, Embeddings
from openai import OpenAI

from config import CHROMA_PERSIST_DIR, QWEN_API_KEY, QWEN_BASE_URL, RAG_TOP_K

EMBEDDING_MODEL_NAME = "text-embedding-v2"
EMBEDDING_BATCH_SIZE = 25

# ------------------------------------------------------------------
# Embedding Function（不变）
# ------------------------------------------------------------------

class DashScopeEmbeddingFunction:
    def __init__(self):
        self._client = OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)

    def embed_documents(self, input: Documents) -> Embeddings:
        return self(input)

    def embed_query(self, text: str) -> list[float]:
        result = self([text])
        return result[0] if result else []

    def __call__(self, input: Documents) -> Embeddings:
        if not input:
            return []
        all_embeddings = []
        for i in range(0, len(input), EMBEDDING_BATCH_SIZE):
            batch = input[i:i + EMBEDDING_BATCH_SIZE]
            try:
                resp = self._client.embeddings.create(
                    model=EMBEDDING_MODEL_NAME,
                    input=batch,
                )
                all_embeddings.extend([d.embedding for d in resp.data])
            except Exception as e:
                print(f"[RAG] Embedding batch failed: {e}")
                dim = 1536
                all_embeddings.extend([[0.0] * dim for _ in batch])
        return all_embeddings


_embedding_fn = DashScopeEmbeddingFunction()


# ------------------------------------------------------------------
# ChromaDB 工具方法
# ------------------------------------------------------------------

def _get_collection_safe(client, name, ef):
    try:
        return client.get_collection(name=name, embedding_function=ef)
    except Exception:
        try:
            return client.get_collection(name=name)
        except Exception:
            return None


def _build_doc_text(item: dict, library: str) -> str:
    """将不同库的条目转为用于 embedding 的统一文本表示"""
    if library == "red_lines":
        return f"{item.get('source','')} {item.get('category','')} {item.get('content','')} {' '.join(item.get('keywords',[]))}"
    elif library == "black_words":
        return (
            f"黑词: {item.get('word','')} 分类: {item.get('category','')} "
            f"原事件: {item.get('original_incident','')} "
            f"致命句式: {item.get('fatal_sentence_pattern','')} "
            f"爆炸原因: {item.get('why_it_exploded','')} "
            f"语境说明: {item.get('context_matters','')}"
        )
    elif library == "hate_comments":
        return f"恶评: {item.get('content','')} 情绪: {item.get('emotion_type','')} 修辞: {item.get('rhetoric_style','')}"
    elif library == "case_studies":
        return (
            f"案例: {item.get('title','')} 类型: {item.get('crisis_type','')} "
            f"描述: {item.get('description','')} 公众反应: {item.get('public_reaction','')}"
        )
    return ""


# ------------------------------------------------------------------
# 分类框架加载器（纯 JSON，不走向量检索）
# ------------------------------------------------------------------

def _load_json_kb(data_dir: str, filename: str) -> dict:
    """加载 JSON 知识库文件"""
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        print(f"[RAG] 警告: {filename} 不存在，跳过")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------
# 关键词匹配引擎（核心创新）
# ------------------------------------------------------------------

def _keyword_hit_score(text: str, keywords: list[str]) -> float:
    """计算文本对关键词列表的命中率（0-1）"""
    if not keywords or not text:
        return 0.0
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    return hits / len(keywords)


def match_taxonomy_patterns(event_desc: str, taxonomy: list[dict], top_k: int = 5) -> list[dict]:
    """
    基于关键词 + 语义相似度的混合匹配，找到最相关的危机模式。

    返回格式:
    [{
        "pattern_id": "p01_sub01",
        "category": "责任推卸",
        "subtype_name": "甩锅给临时工/外包人员",
        "keyword_score": 0.67,
        "risk_level": 5,
        "thinking_prompt": "...",
        "danger_reason": "...",
        "template_phrases": [...],
        "matched_keywords": ["外包", "第三方"]
    }, ...]
    """
    if not taxonomy or not event_desc:
        return []

    results = []
    for pattern in taxonomy:
        for subtype in pattern.get("subtypes", []):
            keywords = subtype.get("keywords", [])
            kscore = _keyword_hit_score(event_desc, keywords)

            if kscore > 0:  # 至少命中一个关键词
                results.append({
                    "pattern_id": subtype["id"],
                    "category": pattern["category"],
                    "category_tag": pattern.get("category_tag", ""),
                    "core_angle": pattern.get("core_angle", ""),
                    "subtype_name": subtype["name"],
                    "keyword_score": round(kscore, 3),
                    "risk_level": subtype.get("risk_level", 3),
                    "thinking_prompt": subtype.get("thinking_prompt", ""),
                    "danger_reason": subtype.get("danger_reason", ""),
                    "template_phrases": subtype.get("template_phrases", []),
                    "matched_keywords": [kw for kw in keywords if kw.lower() in event_desc.lower()],
                })

    # 按 keyword_score × risk_level 排序
    results.sort(key=lambda x: x["keyword_score"] * x["risk_level"], reverse=True)
    return results[:top_k]


def match_keyword_templates(event_desc: str, templates: list[dict], top_k: int = 3) -> list[dict]:
    """
    基于关键词组合匹配场景模板。

    返回格式:
    [{
        "template_id": "tpl_service_abuse",
        "scenario_type": "服务人员辱骂/威胁消费者",
        "keyword_score": 0.67,
        "maps_to_patterns": ["p01_sub01", "p02_sub02"],
        "auto_generated_scenario": "...",
        "thinking_guide": "..."
    }, ...]
    """
    if not templates or not event_desc:
        return []

    results = []
    for tpl in templates:
        combos = tpl.get("keyword_combos", [])
        best_hit = 0.0
        best_matched = []
        for combo in combos:
            hits = [kw for kw in combo if kw.lower() in event_desc.lower()]
            if len(hits) >= 2:  # 至少命中2个关键词才算模板匹配
                score = len(hits) / len(combo)
                if score > best_hit:
                    best_hit = score
                    best_matched = hits

        if best_hit > 0:
            results.append({
                "template_id": tpl["id"],
                "scenario_type": tpl["scenario_type"],
                "keyword_score": round(best_hit, 3),
                "maps_to_patterns": tpl.get("maps_to_patterns", []),
                "auto_generated_scenario": tpl.get("auto_generated_scenario", ""),
                "thinking_guide": tpl.get("thinking_guide", ""),
                "matched_keywords": best_matched,
                "elements": tpl.get("elements", {}),
            })

    results.sort(key=lambda x: x["keyword_score"], reverse=True)
    return results[:top_k]


def get_checklist_for_role(checklist_data: dict, agent_role: str = "official") -> list[dict]:
    """
    根据 Agent 角色获取对应的审查清单。

    agent_role: "official" | "influencer" | "bystander"
    返回: 该角色适用的阶段和清单项列表
    """
    phases = checklist_data.get("phases", [])
    role_config = checklist_data.get("_role_assignment", {}).get(agent_role, {})
    applicable_phases = role_config.get("phases", [1, 2])

    result = []
    for phase in phases:
        if phase["phase"] in applicable_phases:
            result.append({
                "phase": phase["phase"],
                "title": phase["title"],
                "description": phase["description"],
                "items": phase["items"],
            })
    return result


# ------------------------------------------------------------------
# RAG 引擎（v2.0）
# ------------------------------------------------------------------

class RAGEngine:
    """多库联合检索引擎 — v2.0 融合分类框架 + 关键词模板"""

    def __init__(self, data_dir: str = "data", persist_dir: str = CHROMA_PERSIST_DIR):
        os.makedirs(persist_dir, exist_ok=True)
        self._client = PersistentClient(path=persist_dir)
        self._ef = _embedding_fn
        self._collections = {}
        self._data_dir = data_dir

        # v2.0: 加载逻辑知识库
        self._taxonomy = None
        self._templates = None
        self._checklist = None
        self._ensure_logical_kb_loaded()

    def _ensure_logical_kb_loaded(self):
        """懒加载三个逻辑知识库 JSON"""
        if self._taxonomy is None:
            raw = _load_json_kb(self._data_dir, "crisis_taxonomy.json")
            self._taxonomy = raw.get("taxonomy", [])
        if self._templates is None:
            raw = _load_json_kb(self._data_dir, "keyword_templates.json")
            self._templates = raw.get("templates", [])
        if self._checklist is None:
            self._checklist = _load_json_kb(self._data_dir, "audit_checklist.json")

    # ------------------------------------------------------------------
    # ChromaDB 索引构建（不变）
    # ------------------------------------------------------------------

    def build_all(self, data_dir: str = "data"):
        self.build_index("red_lines", os.path.join(data_dir, "red_line_library.json"))
        self.build_index("black_words", os.path.join(data_dir, "black_words_library.json"))
        self.build_index("hate_comments", os.path.join(data_dir, "hate_comments_library.json"))
        self.build_index("case_studies", os.path.join(data_dir, "case_studies.json"))

    def build_index(self, library: str, json_path: str):
        if not os.path.exists(json_path):
            print(f"[RAG] 跳过 {library}: 文件不存在 {json_path}")
            return
        with open(json_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        if not isinstance(items, list) or not items:
            print(f"[RAG] 跳过 {library}: 数据为空")
            return
        try:
            self._client.delete_collection(name=library)
        except Exception:
            pass
        collection = self._client.create_collection(
            name=library,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )
        docs, ids, metadatas = [], [], []
        for i, item in enumerate(items):
            doc_text = _build_doc_text(item, library)
            if not doc_text.strip():
                continue
            docs.append(doc_text)
            ids.append(item.get("id", f"{library}_{i}"))
            meta = {}
            if library == "red_lines":
                meta = {
                    "source": str(item.get("source", ""))[:200],
                    "category": str(item.get("category", ""))[:100],
                    "severity": item.get("severity", 3),
                    "penalty": str(item.get("penalty", ""))[:200],
                }
            elif library == "black_words":
                meta = {
                    "word": str(item.get("word", ""))[:100],
                    "category": str(item.get("category", ""))[:100],
                    "why_it_exploded": str(item.get("why_it_exploded", ""))[:300],
                    "fatal_sentence_pattern": str(item.get("fatal_sentence_pattern", ""))[:200],
                    "context_matters": str(item.get("context_matters", ""))[:200],
                }
            elif library == "hate_comments":
                meta = {
                    "content": str(item.get("content", ""))[:500],
                    "emotion_type": str(item.get("emotion_type", ""))[:100],
                    "rhetoric_style": str(item.get("rhetoric_style", ""))[:200],
                    "why_effective": str(item.get("why_effective", ""))[:300],
                }
            elif library == "case_studies":
                meta = {
                    "title": str(item.get("title", ""))[:200],
                    "crisis_type": str(item.get("crisis_type", ""))[:200],
                    "public_reaction": str(item.get("public_reaction", ""))[:400],
                }
            metadatas.append(meta)
        if docs:
            collection.add(documents=docs, ids=ids, metadatas=metadatas)
        self._collections[library] = collection
        print(f"[RAG] {library}: 索引 {len(docs)} 条记录")

    # ------------------------------------------------------------------
    # ChromaDB 检索（不变）
    # ------------------------------------------------------------------

    def _get_collection(self, library: str):
        if library not in self._collections:
            col = _get_collection_safe(self._client, library, self._ef)
            if col is None:
                return None
            self._collections[library] = col
        return self._collections[library]

    def search(self, library: str, query: str, top_k: int = RAG_TOP_K):
        col = self._get_collection(library)
        if col is None:
            return []
        try:
            results = col.query(query_texts=[query], n_results=top_k)
            if not results or not results.get("ids") or not results["ids"][0]:
                return []
            out = []
            for i, doc_id in enumerate(results["ids"][0]):
                dist = results["distances"][0][i] if results.get("distances") else 1.0
                meta = results["metadatas"][0][i] if results.get("metadatas") else {}
                out.append({
                    "id": doc_id,
                    "distance": round(dist, 4),
                    "similarity": round(1 - dist, 4),
                    **meta,
                })
            return out
        except Exception as e:
            print(f"[RAG] 检索 {library} 失败: {e}")
            return []

    # ------------------------------------------------------------------
    # v2.0 核心: 分类框架匹配 + 审查清单
    # ------------------------------------------------------------------

    def match_patterns(self, event_desc: str) -> list[dict]:
        """
        基于关键词在 taxonomy 中匹配危机模式。
        这是 v2.0 的核心方法——用思考框架替代案例参考。
        """
        self._ensure_logical_kb_loaded()
        return match_taxonomy_patterns(event_desc, self._taxonomy)

    def match_templates(self, event_desc: str) -> list[dict]:
        """基于关键词组合匹配场景模板"""
        self._ensure_logical_kb_loaded()
        return match_keyword_templates(event_desc, self._templates)

    def get_checklist(self, agent_role: str = "official") -> list[dict]:
        """获取指定 Agent 角色的审查清单"""
        self._ensure_logical_kb_loaded()
        return get_checklist_for_role(self._checklist, agent_role)

    # ------------------------------------------------------------------
    # 联合检索（旧接口兼容 + 新框架）
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = RAG_TOP_K):
        """联合检索 4 个 ChromaDB 向量库"""
        return {
            "red_lines": self.search("red_lines", query, top_k),
            "black_words": self.search("black_words", query, top_k),
            "hate_comments": self.search("hate_comments", query, top_k),
            "similar_cases": self.search("case_studies", query, top_k),
        }

    def retrieve_v2(self, event_desc: str, agent_role: str = "official") -> dict:
        """
        v2.0 联合检索：向量检索 + 分类框架匹配 + 审查清单。
        返回完整结构，供 Prompt 注入使用。
        """
        # ChromaDB 向量检索（保留）
        vector_results = self.retrieve(event_desc)

        # v2.0 分类框架匹配
        pattern_matches = self.match_patterns(event_desc)
        template_matches = self.match_templates(event_desc)

        # v2.0 审查清单
        checklist = self.get_checklist(agent_role)

        return {
            # 旧接口兼容
            "red_lines": vector_results["red_lines"],
            "black_words": vector_results["black_words"],
            "hate_comments": vector_results["hate_comments"],
            "similar_cases": vector_results["similar_cases"],
            # v2.0 新增
            "pattern_matches": pattern_matches,
            "template_matches": template_matches,
            "audit_checklist": checklist,
        }

    def retrieve_similar_case(self, query: str):
        """兼容旧接口"""
        cases = self.search("case_studies", query, top_k=1)
        if not cases:
            return None
        best = cases[0]
        return {
            "id": best["id"],
            "title": best.get("title", ""),
            "crisis_type": best.get("crisis_type", ""),
            "public_reaction": best.get("public_reaction", ""),
        }

    # ------------------------------------------------------------------
    # v2.0 Prompt 格式化输出
    # ------------------------------------------------------------------

    def format_taxonomy_context(self, pattern_matches: list[dict]) -> str:
        """
        将匹配到的危机模式格式化为可注入 Agent prompt 的思考框架。

        输出示例:
        【危机模式匹配 · 思考框架】
        ■ 模式 #1: 责任推卸 → 甩锅给临时工/外包人员 (风险等级: 5/5)
          思考角度: 这个声明是否在把责任推给'临时工'或'外包'？
          高危信号: 公众早已识破此套路，使用即触发群嘲
          禁用措辞: 系第三方外包公司人员 / 并非我司正式员工
        ■ 模式 #2: ...
        """
        if not pattern_matches:
            return ""

        lines = ["【危机模式匹配 · 思考框架】"]
        for i, pm in enumerate(pattern_matches, 1):
            lines.append(
                f"■ 模式 #{i}: {pm['category']} → {pm['subtype_name']} "
                f"(风险等级: {pm['risk_level']}/5, 关键词命中: {pm.get('matched_keywords', [])})"
            )
            if pm.get("thinking_prompt"):
                lines.append(f"  🧠 思考角度: {pm['thinking_prompt']}")
            if pm.get("danger_reason"):
                lines.append(f"  ⚠️ 高危信号: {pm['danger_reason']}")
            if pm.get("template_phrases"):
                phrases = " / ".join(pm['template_phrases'][:3])
                lines.append(f"  🚫 禁用措辞: {phrases}")
            if pm.get("core_angle"):
                lines.append(f"  📐 审查主轴: {pm['core_angle']}")
            lines.append("")

        return "\n".join(lines)

    def format_template_context(self, template_matches: list[dict]) -> str:
        """将场景模板匹配结果格式化为 prompt 注入"""
        if not template_matches:
            return ""
        lines = ["【场景模板匹配 · 关键词触发】"]
        for i, tm in enumerate(template_matches, 1):
            lines.append(
                f"□ 场景 #{i}: {tm['scenario_type']} "
                f"(关键词: {tm.get('matched_keywords', [])})"
            )
            if tm.get("thinking_guide"):
                lines.append(f"  💡 场景思考: {tm['thinking_guide']}")
            lines.append("")
        return "\n".join(lines)

    def format_checklist_context(self, checklist: list[dict]) -> str:
        """
        将审查清单格式化为 Agent 的 TODO 列表。

        输出示例:
        【审查清单 · 第1阶段: 事件定性】
        □ [todo_001] 识别危机类型：是产品问题？态度问题？还是价值观冒犯？
        □ [todo_002] 判断责任归属：品牌方是否有错？错在哪里？
        ...
        """
        if not checklist:
            return ""
        lines = ["【审查清单 · 请逐项完成以下思考】"]
        for phase_info in checklist:
            lines.append(
                f"\n--- 阶段 {phase_info['phase']}: {phase_info['title']} ---"
            )
            lines.append(f"（{phase_info.get('description', '')}）")
            for item in phase_info.get("items", []):
                lines.append(f"  □ [{item['id']}] {item['task']}")
                if item.get("guide"):
                    lines.append(f"     💡 {item['guide']}")
        lines.append("\n【重要】请在你的输出中逐一回应以上清单项，标注对应的编号。")
        return "\n".join(lines)

    def format_context_for_agent(self, result: dict) -> str:
        """v2.0: 将检索结果格式化为可直接注入 Agent prompt 的上下文块"""
        parts = []

        # ---- v2.0 优先: 分类框架 + 审查清单 ----
        pattern_matches = result.get("pattern_matches", [])
        if pattern_matches:
            taxonomy_ctx = self.format_taxonomy_context(pattern_matches)
            if taxonomy_ctx:
                parts.append(taxonomy_ctx)

        template_matches = result.get("template_matches", [])
        if template_matches:
            template_ctx = self.format_template_context(template_matches)
            if template_ctx:
                parts.append(template_ctx)

        checklist = result.get("audit_checklist", [])
        if checklist:
            checklist_ctx = self.format_checklist_context(checklist)
            if checklist_ctx:
                parts.append(checklist_ctx)

        # ---- v1.0 兼容: 红线/黑词/案例 ----
        red_lines = result.get("red_lines", [])
        if red_lines:
            lines = []
            for rl in red_lines[:3]:
                lines.append(
                    f"• {rl.get('source','')} (严重度:{rl.get('severity','?')}/5): "
                    f"{rl.get('category','')} — {rl.get('penalty','')}"
                )
            parts.append(f"【法规红线参考】\n" + "\n".join(lines))

        black_words = result.get("black_words", [])
        if black_words:
            lines = []
            for bw in black_words[:3]:
                lines.append(
                    f"• 「{bw.get('word','')}」— {bw.get('why_it_exploded','')[:150]}"
                )
            parts.append(f"【敏感词预警】\n" + "\n".join(lines))

        return "\n\n".join(parts)

    def format_master_agent_context(self, result: dict) -> str:
        """
        Master Agent 专属上下文格式化 —— 更侧重策略框架而非逐项清单。
        Master Agent 需要的是：危机模式概览 + 角色配置建议。
        """
        parts = []

        pattern_matches = result.get("pattern_matches", [])
        if pattern_matches:
            lines = ["## 危机模式识别"]
            for pm in pattern_matches:
                lines.append(
                    f"- **{pm['category']} / {pm['subtype_name']}** "
                    f"(风险 {pm['risk_level']}/5, 关键词: {pm.get('matched_keywords', [])})"
                )
                if pm.get("core_angle"):
                    lines.append(f"  → {pm['core_angle'][:120]}")
            parts.append("\n".join(lines))

        template_matches = result.get("template_matches", [])
        if template_matches:
            lines = ["## 场景模板匹配"]
            for tm in template_matches:
                lines.append(
                    f"- **{tm['scenario_type']}** (命中词: {tm.get('matched_keywords', [])})"
                )
                if tm.get("thinking_guide"):
                    lines.append(f"  → {tm['thinking_guide'][:150]}")
            parts.append("\n".join(lines))

        return "\n\n".join(parts)


# ------------------------------------------------------------------
# 全局单例
# ------------------------------------------------------------------
_engine: RAGEngine | None = None


def get_rag_engine() -> RAGEngine:
    global _engine
    if _engine is None:
        _engine = RAGEngine()
    return _engine


def ensure_indexes(data_dir: str = "data"):
    """确保 ChromaDB 向量索引已构建"""
    engine = get_rag_engine()
    if engine._get_collection("case_studies") is None:
        print("[RAG] 首次启动，构建所有向量索引...")
        engine.build_all(data_dir)
    else:
        for lib in ["red_lines", "black_words", "hate_comments", "case_studies"]:
            engine._get_collection(lib)
