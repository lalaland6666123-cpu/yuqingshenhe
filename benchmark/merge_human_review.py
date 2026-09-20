"""Merge the pasted human review CSV into test.jsonl without losing raw labels."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LEVEL = {"low": 1, "medium": 3, "high": 5}


def split_domains(text: str) -> list[str]:
    text = (text or "").replace("&", "and")
    return [x.strip() for x in text.split("and") if x.strip()]


def main(csv_path: str) -> None:
    review = {}
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            review[row["id"]] = row

    test_path = ROOT / "test.jsonl"
    merged = []
    missing = []
    for line in test_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        row = review.get(item["id"])
        if row is None:
            missing.append(item["id"])
            merged.append(item)
            continue
        scores = {k: LEVEL.get((row[f"human_{k}"] or "").strip().lower()) for k in ("legal", "business", "reputation")}
        item["human_annotation"] = {
            "risk_spans": [],
            "crisis_types": split_domains(row["human_crisis_types"]),
            "risk_scores": scores,
            "gold_actions": [row["human_actions"].strip()] if row["human_actions"].strip() else [],
            "failure_tags": [],
            "decision": row["human_actions"].strip().lower(),
            "raw": row,
        }
        item["annotation_status"] = "human_reviewed_score_only"
        merged.append(item)

    out = ROOT / "test_reviewed.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for item in merged:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    reviewed = [x for x in merged if x.get("human_annotation")]
    decisions = {"approve": 0, "reject": 0}
    score_counts = {k: {str(i): 0 for i in (1, 3, 5)} for k in ("legal", "business", "reputation")}
    pre_decision_matches = 0
    for item in reviewed:
        ann = item["human_annotation"]
        decision = ann["decision"]
        if decision in decisions:
            decisions[decision] += 1
        for dim, score in ann["risk_scores"].items():
            if score is not None:
                score_counts[dim][str(score)] += 1
        pre = item["preannotation"].get("risk_scores", {}).get("reputation", 1)
        pre_is_risky = pre >= 3
        human_is_risky = decision == "reject"
        pre_decision_matches += int(pre_is_risky == human_is_risky)
    report = {
        "num_test_cases": len(merged),
        "num_human_reviewed": len(reviewed),
        "missing_review_ids": missing,
        "decision_counts": decisions,
        "score_counts": score_counts,
        "preannotation_reputation_vs_human_decision_agreement": pre_decision_matches / max(1, len(reviewed)),
        "mapping": "Low=1, Medium=3, High=5; raw labels preserved under human_annotation.raw",
        "limitation": "No span offsets were supplied, so span-level F1 is not computed.",
    }
    report_path = ROOT / "human_review_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python merge_human_review.py REVIEW.csv")
    main(sys.argv[1])
