# Evaluation

`run_rule_baseline.py` 是第一版可复现基线，不调用外部 API。它用于验证数据格式、指标计算和实验流水线是否连通，不代表最终系统性能。

后续基线应保持同一输入和测试集：BM25、Embedding-only、Single-LLM、Full System，并在 `results/` 保存机器可读 JSON。所有结果必须记录数据版本、模型、温度、提示版本和标签来源。

当前候选集仍使用 `preannotation`，结果只能作为流水线 smoke test。人工审核后，脚本会自动优先读取 `human_annotation`。
