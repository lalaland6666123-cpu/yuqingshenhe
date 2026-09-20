"""Generate a reviewable candidate set for the Chinese PR risk benchmark.

The generated labels are weak/pre-labels derived from controlled templates. They
must be reviewed by humans before being reported as benchmark ground truth.
"""
from __future__ import annotations

import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent

SCENARIOS = [
    ("blame_shifting", "责任推诿", "把问题归咎于外包人员或个别员工", "我们已终止与涉事外包人员的合作，相关问题与公司无关。"),
    ("arrogance", "傲慢冒犯", "以居高临下的口吻评价消费者", "真正懂产品的人不会因为这点小问题抱怨，请大家理性一点。"),
    ("deception", "虚假宣传", "使用无法证明的绝对化承诺", "我们的产品百分之百安全有效，任何人都可以放心使用。"),
    ("indifference", "冷漠拖延", "使用模板化话术回避具体处理", "我们高度重视相关反馈，后续将持续关注并适时发布消息。"),
    ("discrimination", "歧视与排斥", "把特定群体描述为不配使用产品", "这款产品只面向高端人群，普通消费者没有必要了解。"),
    ("privacy", "隐私与数据", "在公开回应中暴露或索取用户隐私", "请把身份证号和完整订单截图发到评论区，我们才能核实。"),
    ("safety", "安全与质量", "淡化可能造成伤害的质量或安全问题", "目前只是极少数个案，大家不必过度解读，产品总体没有问题。"),
    ("governance", "治理与合规", "承诺删除评论或用强硬手段压制反馈", "我们会清理所有负面评论，对持续传播者保留追责权利。"),
]

SAFE_SUFFIXES = [
    "我们会在今天18点前公布核查范围、负责人和下一次更新时间。",
    "在事实确认前不会推测原因，也不会公开任何用户个人信息。",
    "如核查确认存在问题，我们将说明影响范围、补救措施和完成时间。",
]


def make_record(index: int, scenario, risky: bool, variant: int) -> dict:
    tag, name, rationale, risky_draft = scenario
    if risky:
        draft = risky_draft
        span = risky_draft
        severity = 4 if tag in {"privacy", "safety", "deception"} else 3
        actions = ["删除或改写触发风险的句子", "说明具体事实和处理时限", "保留可核验的证据"]
    else:
        draft = SAFE_SUFFIXES[variant % len(SAFE_SUFFIXES)]
        span = ""
        severity = 1
        actions = ["继续核实事实", "公开下一次更新时间", "避免披露个人信息"]
    return {
        "id": f"case_{index:04d}",
        "source": "synthetic_template",
        "annotation_status": "preannotated_needs_review",
        "scenario": tag,
        "scenario_name": name,
        "event_description": f"某品牌收到关于{rationale}的集中投诉，相关内容在社交平台快速传播。",
        "draft_response": draft,
        "preannotation": {
            "risk_spans": ([{"text": span, "type": name, "severity": severity}] if risky else []),
            "crisis_types": [tag] if risky else [],
            "risk_scores": {"legal": severity, "business": severity, "reputation": severity},
            "gold_actions": actions,
            "failure_tags": [],
        },
        "human_annotation": None,
    }


def main() -> None:
    random.seed(20260920)
    records = []
    idx = 1
    for scenario in SCENARIOS:
        for variant in range(30):
            # 20 risky and 10 safe examples per category.
            records.append(make_record(idx, scenario, variant < 20, variant))
            idx += 1
    random.shuffle(records)
    test = records[:48]
    dev = records[48:96]
    train = records[96:]
    for name, rows in (("cases", records), ("train", train), ("dev", dev), ("test", test)):
        with (ROOT / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"generated {len(records)} candidates: train={len(train)} dev={len(dev)} test={len(test)}")


if __name__ == "__main__":
    main()
