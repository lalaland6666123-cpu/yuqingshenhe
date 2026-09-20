# Benchmark

这是 PR Wind Tunnel 的独立评测集。运行 `python generate_candidates.py` 会生成 240 条候选样本，并划分为 train/dev/test。当前样本是模板化候选集，必须经过人工复核后才能作为正式测试集。

推荐评测任务：危机类型分类、风险片段识别、红线/黑词检索、三维风险评分、处理建议质量和端到端报告质量。测试集按场景拆分，避免同一模板的轻微改写造成泄漏。

目录中的 `schema.json` 定义字段，`annotation_guideline.md` 定义人工标注规则。标注后的数据仍保留 `preannotation`，便于分析弱标注与人工真值之间的差异。
