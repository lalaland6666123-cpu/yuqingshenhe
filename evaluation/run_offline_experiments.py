"""Run reproducible offline proxies when external LLM access is unavailable.

These results are explicitly labelled offline_proxy. They validate the
benchmark/evaluation loop, but must not be presented as DeepSeek results.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEYWORDS = {
    "blame_shifting": ["外包", "无关", "个别员工"],
    "arrogance": ["真正懂", "理性一点"],
    "deception": ["百分之百", "绝对", "放心使用"],
    "indifference": ["高度重视", "持续关注", "适时发布"],
    "discrimination": ["普通消费者", "高端人群"],
    "privacy": ["身份证号", "完整订单", "评论区"],
    "safety": ["极少数个案", "过度解读", "总体没有问题"],
    "governance": ["清理所有负面", "追责权利", "删除评论"],
}
HIGH_LEGAL = {"deception", "privacy", "safety"}


def hits(text):
    return {tag for tag, words in KEYWORDS.items() if any(word in text for word in words)}


def predict(row, mode):
    found = hits(row["draft_response"])
    if mode == "rule_only":
        decision = "reject" if found else "approve"
        return {"decision": decision, "legal": 1 + min(4, len(found)), "business": 1 + min(4, len(found)), "reputation": 1 + min(4, len(found))}
    # The knowledge-augmented proxy uses risk-specific severity priors, standing
    # in for taxonomy/checklist evidence without calling an external model.
    legal = 1 + min(4, sum(2 if x in HIGH_LEGAL else 0 for x in found))
    business = 1 + min(4, len(found) + (1 if "safety" in found else 0))
    reputation = 1 + min(4, len(found) + (1 if found else 0))
    return {"decision": "reject" if found else "approve", "legal": legal, "business": business, "reputation": reputation}


def score(rows, mode):
    decision_hits = 0
    errors = []
    total_mae = 0.0
    for row in rows:
        pred = predict(row, mode)
        truth = row["human_annotation"]
        decision_hits += int(pred["decision"] == truth["decision"])
        err = sum(abs(pred[k] - truth["risk_scores"][k]) for k in ("legal", "business", "reputation")) / 3
        total_mae += err
        if pred["decision"] != truth["decision"] or err > 1.0:
            errors.append({"id": row["id"], "prediction": pred, "truth": truth["risk_scores"], "draft": row["draft_response"]})
    return {"mode": mode, "type": "offline_proxy", "num_cases": len(rows), "decision_accuracy": decision_hits / max(1, len(rows)), "mean_score_mae": total_mae / max(1, len(rows)), "errors": errors}


def main():
    rows = [json.loads(x) for x in (ROOT / "benchmark" / "test_reviewed.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    results = {
        "rule_only": score(rows, "rule_only"),
        "rag_augmented_proxy": score(rows, "rag_augmented_proxy"),
        "full_system_proxy": score(rows, "rag_augmented_proxy"),
    }
    out = ROOT / "evaluation" / "results" / "offline_experiments.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    report = ROOT / "evaluation" / "EXPERIMENTS.md"
    lines = [
        "# 最小闭环实验记录",
        "",
        "本次结果来自 48 条人工审核测试样本。由于当前网络无法连接 DeepSeek，以下为可复现的离线代理实验，不应标注为 LLM 真实结果。",
        "",
        "| 系统 | 类型 | 决策准确率 | 三维风险分数 MAE |",
        "|---|---|---:|---:|",
    ]
    lines.extend(f"| {k} | {v['type']} | {v['decision_accuracy']:.3f} | {v['mean_score_mae']:.3f} |" for k, v in results.items())
    lines.extend([
        "",
        "消融定义：`rule_only` 只使用触发词；`rag_augmented_proxy` 使用风险类别和维度先验模拟知识库/审核清单增强；`full_system_proxy` 当前与增强代理共享离线决策器，真实多智能体系统需在 API 可用后重新运行。",
        "",
        "真实运行入口：`run_deepseek_baselines.py`。",
        "",
    ])
    report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: {x: v[x] for x in ("decision_accuracy", "mean_score_mae")} for k, v in results.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
