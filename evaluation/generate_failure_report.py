"""Turn offline mismatch records into a concise, reviewable failure report."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / "evaluation" / "results" / "offline_experiments.json").read_text(encoding="utf-8"))
rows = data["rag_augmented_proxy"]["errors"]
lines = [
    "# Failure Cases",
    "",
    "以下案例来自人工审核测试集，按离线知识库增强代理的分数误差或决策错误筛选。它们用于回归测试，不代表真实 LLM 输出。",
    "",
]
for i, row in enumerate(rows[:10], 1):
    lines.extend([
        f"## FC-{i:02d} {row['id']}",
        "",
        f"- 文案：{row['draft']}",
        f"- 预测：{row['prediction']}",
        f"- 人工分数：{row['truth']}",
        "- 初步原因：规则能够发现表面风险，但无法准确估计不同风险维度的影响范围或严重程度。",
        "- 修复方向：增加逐维度标注、相似案例证据和高风险对抗样本，并在报告中要求证据与分数逐项对应。",
        "",
    ])
(ROOT / "evaluation" / "FAILURE_CASES.md").write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {min(10, len(rows))} failure cases")
