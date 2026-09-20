"""Run direct LLM and local-taxonomy-augmented baselines.

This script uses only synthetic benchmark text. It writes raw predictions so
every reported number is auditable. Set DEEPSEEK_API_KEY before running.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from openai import OpenAI

from metrics import mae

ROOT = Path(__file__).resolve().parents[1]
TAGS = ["blame_shifting", "arrogance", "deception", "indifference", "discrimination", "privacy", "safety", "governance"]


def load_taxonomy():
    path = ROOT / ".." / "data" / "crisis_taxonomy.json"
    if not path.exists():
        path = Path(r"C:\Users\33395\Desktop\舆情风控审核\data\crisis_taxonomy.json")
    if not path.exists():
        return []
    obj = json.loads(path.read_text(encoding="utf-8"))
    return obj.get("taxonomy", [])


def retrieve_guidance(text: str, taxonomy) -> str:
    hits = []
    for category in taxonomy:
        for subtype in category.get("subtypes", []):
            words = subtype.get("keywords", [])
            score = sum(1 for word in words if word and word in text)
            if score:
                hits.append((score * subtype.get("risk_level", 1), category.get("category", ""), subtype.get("name", ""), subtype.get("thinking_prompt", "")))
    hits.sort(reverse=True)
    return "\n".join(f"- {x[1]} / {x[2]}: {x[3]}" for x in hits[:3])


def parse_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text or "").strip().strip("`")
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def call(client, event: str, draft: str, guidance: str = "") -> dict:
    context = f"\n本地知识库检索提示：\n{guidance}" if guidance else ""
    prompt = f"""你是中文舆情风控审核员。只根据输入判断，不补充外部事实。
事件：{event}
待审文案：{draft}{context}
请只输出 JSON：{{\"decision\":\"approve 或 reject\",\"legal\":1,\"business\":1,\"reputation\":1,\"reason\":\"一句话\"}}。
分数必须是 1 到 5 的整数。"""
    response = client.chat.completions.create(
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        temperature=0,
        max_tokens=180,
        messages=[{"role": "system", "content": "你是严谨的结构化文本审核器。"}, {"role": "user", "content": prompt}],
    )
    return parse_json(response.choices[0].message.content)


def evaluate(rows, predictions):
    decision_correct = score_errors = 0
    errors = []
    for row, pred in zip(rows, predictions):
        ann = row["human_annotation"]
        truth_decision = ann["decision"]
        pred_decision = str(pred.get("decision", "")).lower()
        decision_correct += int(pred_decision == truth_decision)
        truth_scores = ann["risk_scores"]
        vals = [int(pred.get(k, 3)) for k in ("legal", "business", "reputation") if str(pred.get(k, "")).isdigit()]
        score_errors += sum(abs(int(truth_scores[k]) - int(pred.get(k, 3))) for k in ("legal", "business", "reputation")) / 3
        if pred_decision != truth_decision or not vals:
            errors.append({"id": row["id"], "truth": ann, "prediction": pred, "draft": row["draft_response"]})
    return {"decision_accuracy": decision_correct / max(1, len(rows)), "mean_score_mae": score_errors / max(1, len(rows)), "num_cases": len(rows), "errors": errors}


def main(path: str, limit: int | None = None):
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key or key.startswith("your-"):
        raise SystemExit("DEEPSEEK_API_KEY is missing or is a placeholder")
    rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    if limit:
        rows = rows[:limit]
    client = OpenAI(api_key=key, base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    taxonomy = load_taxonomy()
    outputs = {}
    for name, augmented in (("single_llm", False), ("rag_augmented_llm", True)):
        preds = []
        for i, row in enumerate(rows, 1):
            guidance = retrieve_guidance(row["event_description"] + " " + row["draft_response"], taxonomy) if augmented else ""
            try:
                pred = call(client, row["event_description"], row["draft_response"], guidance)
            except Exception as exc:
                pred = {"error": str(exc)}
            preds.append(pred)
            print(f"{name}: {i}/{len(rows)}", flush=True)
        outputs[name] = {"metrics": evaluate(rows, preds), "predictions": preds}
    out = ROOT / "evaluation" / "results" / "deepseek_baselines.json"
    out.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v["metrics"] | {"errors": len(v["metrics"]["errors"])} for k, v in outputs.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "benchmark" / "test_reviewed.jsonl"), int(sys.argv[2]) if len(sys.argv) > 2 else None)
