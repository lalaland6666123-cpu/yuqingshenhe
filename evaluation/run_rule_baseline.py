"""Run a transparent keyword/rule baseline on the candidate benchmark."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from metrics import mae, precision_recall_f1, set_f1

ROOT = Path(__file__).resolve().parents[1]
KEYWORDS = {
    "blame_shifting": ["外包", "无关", "个别员工"],
    "arrogance": ["真正懂", "理性一点", "高端人群"],
    "deception": ["百分之百", "绝对", "放心使用"],
    "indifference": ["高度重视", "持续关注", "适时发布"],
    "discrimination": ["普通消费者", "不配", "高端人群"],
    "privacy": ["身份证号", "完整订单", "评论区"],
    "safety": ["极少数个案", "过度解读", "总体没有问题"],
    "governance": ["清理所有负面", "追责权利", "删除评论"],
}


def predict(text):
    hits = {k for k, words in KEYWORDS.items() if any(w in text for w in words)}
    return sorted(hits)


def main(path: str = str(ROOT / "benchmark" / "test.jsonl")):
    rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    truth, pred, true_scores, pred_scores = [], [], [], []
    for row in rows:
        ann = row.get("human_annotation") or row["preannotation"]
        truth.append(ann.get("crisis_types", []))
        pred.append(predict(row["draft_response"]))
        true_scores.append(ann.get("risk_scores", {}).get("reputation", 1))
        pred_scores.append(min(5, max(1, 1 + len(pred[-1]))))
    labels = sorted({x for values in truth + pred for x in values})
    y_true = [values[0] if values else "none" for values in truth]
    y_pred = [values[0] if values else "none" for values in pred]
    comparable = bool(set(y_true) & set(y_pred))
    decision_true = [x.get("human_annotation", {}).get("decision", "").lower() == "reject" for x in rows]
    decision_pred = [bool(x) for x in pred]
    result = {
        "baseline": "rule_only",
        "dataset": str(path),
        "label_source": "human_annotation if present else preannotation",
        "classification": precision_recall_f1(y_true, y_pred) if comparable else None,
        "classification_note": "Skipped: human_crisis_types are broad domains (PR, Legal, Safety), while the rule baseline predicts scenario tags.",
        "multi_label": set_f1(truth, pred) if comparable else None,
        "decision_accuracy": sum(a == b for a, b in zip(decision_true, decision_pred)) / max(1, len(rows)),
        "reputation_score_mae": mae(true_scores, pred_scores),
        "num_cases": len(rows),
        "labels": labels,
    }
    out = ROOT / "evaluation" / "results" / "rule_only_test.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "benchmark" / "test.jsonl"))
