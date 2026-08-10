"""
量化评估体系 — 确定性算法替代 LLM 拍脑袋打分

三个维度：
- Legal:    命中红线条款数量 × 严重程度加权
- Business: 基于 SIR 传播模型的峰值触达人数映射
- Reputation: 历史相似案例后果 + 恶评情感强度综合
"""


def _clamp(score: float, floor: int = 1, ceil: int = 5) -> int:
    return max(floor, min(ceil, round(score)))


# ------------------------------------------------------------------
# 法律风险评分
# ------------------------------------------------------------------

def calculate_legal_score(red_lines: list[dict]) -> dict:
    """
    基于命中的红线条款计算法律风险（1-5）。

    算法：
    - 每条命中红线按其 severity（1-5）加权
    - 命中数 ≥ 5 条直接封顶 5 分
    - 无命中默认 1 分
    """
    if not red_lines:
        return {"score": 1, "reason": "未命中已知红线条款", "hits": 0, "details": []}

    total_severity = sum(rl.get("severity", 3) for rl in red_lines)
    hit_count = len(red_lines)
    max_sev = max(rl.get("severity", 1) for rl in red_lines)

    # 得分 = 命中数 × 平均严重度 的 sigmoid 式压缩到 1-5
    avg_sev = total_severity / hit_count
    raw = avg_sev * min(hit_count, 5) / 3.5
    score = _clamp(raw)

    details = [
        f"{rl.get('source','')} (严重度:{rl.get('severity','?')}/5)"
        for rl in red_lines[:5]
    ]

    return {
        "score": score,
        "reason": f"命中 {hit_count} 条红线条款，最高严重度 {max_sev}/5，加权评分 {score}",
        "hits": hit_count,
        "details": details,
    }


# ------------------------------------------------------------------
# 商业风险评分
# ------------------------------------------------------------------

def calculate_business_score(spread_result) -> dict:
    """
    基于传播模型预测计算商业/销量影响（1-5）。

    算法：
    - 峰值感染人数映射：< 5万→1, 5-20万→2, 20-50万→3, 50-100万→4, >100万→5
    - R₀ > 2 额外 +1
    - 峰值时间早（< t=6）额外 +0.5（早期爆发表明风险更高）
    """
    if spread_result is None:
        return {"score": 2, "reason": "无传播模型数据，默认中等风险"}

    pi = spread_result.peak_infected
    r0 = spread_result.r0
    peak_t = spread_result.peak_time

    if pi < 50_000:
        base = 1
    elif pi < 200_000:
        base = 2
    elif pi < 500_000:
        base = 3
    elif pi < 1_000_000:
        base = 4
    else:
        base = 5

    # R₀ 修正
    if r0 > 3:
        base += 1
    elif r0 > 2:
        base += 0.5

    # 峰值过早修正
    if peak_t < 6:
        base += 0.5

    score = _clamp(base)
    return {
        "score": score,
        "reason": f"预测峰值触达 {pi:,} 人 (t={peak_t})，R₀={r0}，综合评分 {score}",
        "peak_infected": pi,
        "r0": r0,
        "peak_time": peak_t,
    }


# ------------------------------------------------------------------
# 声誉风险评分
# ------------------------------------------------------------------

def calculate_reputation_score(rag_result: dict) -> dict:
    """
    基于历史相似案例 + 恶评情感强度计算声誉损毁度（1-5）。

    算法：
    - 恶评相似度加权（distance 越低越相似，倒数为权重）
    - 情感类型映射：愤怒→3, 讽刺→2.5, 被背叛→2, 自嘲→1.5, 其他→1
    - 历史案例数量归一化
    """
    hate_comments = rag_result.get("hate_comments", []) if rag_result else []
    similar_cases = rag_result.get("similar_cases", []) if rag_result else []

    if not hate_comments and not similar_cases:
        return {"score": 2, "reason": "未检索到高相关度恶评或相似案例，默认中等风险"}

    EMOTION_WEIGHT = {
        "愤怒": 3.0,
        "被背叛感": 2.8,
        "被刺痛": 2.5,
        "讽刺": 2.3,
        "冷嘲": 2.0,
        "厌倦": 1.8,
        "自嘲": 1.5,
    }

    total_weight = 0.0
    count = 0
    for hc in hate_comments:
        sim = max(0.01, hc.get("similarity", 0.5))
        emotion = hc.get("emotion_type", "")
        ew = 1.0
        for key, w in EMOTION_WEIGHT.items():
            if key in emotion:
                ew = w
                break
        total_weight += sim * ew
        count += 1

    avg_hate_weight = total_weight / max(count, 1)
    case_bonus = min(len(similar_cases), 5) * 0.3

    raw = avg_hate_weight * 1.2 + case_bonus
    score = _clamp(raw)

    return {
        "score": score,
        "reason": (
            f"匹配 {len(hate_comments)} 条相关恶评、{len(similar_cases)} 个历史案例，"
            f"恶评情感加权 {avg_hate_weight:.1f}，综合评分 {score}"
        ),
        "hate_comment_count": count,
        "similar_case_count": len(similar_cases),
    }


# ------------------------------------------------------------------
# 综合评估入口
# ------------------------------------------------------------------

def evaluate(event_desc: str, rag_result: dict, spread_result) -> dict:
    """
    综合评估入口，返回与旧 generate_report() 兼容的结构。

    返回格式：
    {
        "fatal_focus": "...",
        "scores": {
            "legal": {"score": 1-5, "reason": "..."},
            "business": {"score": 1-5, "reason": "..."},
            "reputation": {"score": 1-5, "reason": "..."}
        },
        "rewrite_suggestion": "",
        "_deterministic": true
    }
    """
    red_lines = rag_result.get("red_lines", []) if rag_result else []
    black_words = rag_result.get("black_words", []) if rag_result else []
    hate_comments = rag_result.get("hate_comments", []) if rag_result else []
    similar_cases = rag_result.get("similar_cases", []) if rag_result else []

    # 构建 fatal_focus
    focus_parts = []
    if red_lines:
        top_rl = red_lines[0]
        focus_parts.append(f"命中红线：{top_rl.get('source','')}")
    if black_words:
        top_bw = black_words[0]
        focus_parts.append(f"触发黑词「{top_bw.get('word','')}」：{top_bw.get('why_it_exploded','')[:80]}")
    if not focus_parts:
        focus_parts.append("未识别到明确的高危触发点，建议进一步人工审查")

    fatal_focus = "；".join(focus_parts)

    # 构建基于 RAG 证据的改写提示
    rewrite_hints = []
    if red_lines:
        rewrite_hints.append(f"避让 {len(red_lines)} 条法规红线，尤其是 {red_lines[0].get('source','')}")
    if black_words:
        bw_words = [bw.get('word', '') for bw in black_words[:3]]
        rewrite_hints.append(f"严禁使用以下高危措辞：{'、'.join(bw_words)}")
    if similar_cases:
        rewrite_hints.append(f"参考「{similar_cases[0].get('title','')}」的失败教训调整话术")
    rewrite_suggestion = "；".join(rewrite_hints) if rewrite_hints else ""

    return {
        "fatal_focus": fatal_focus,
        "scores": {
            "legal": calculate_legal_score(red_lines),
            "business": calculate_business_score(spread_result),
            "reputation": calculate_reputation_score(rag_result),
        },
        "rewrite_suggestion": rewrite_suggestion,
        "_deterministic": True,
    }
