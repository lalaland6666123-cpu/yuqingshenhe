# 最小闭环实验记录

本次结果来自 48 条人工审核测试样本。由于当前网络无法连接 DeepSeek，以下为可复现的离线代理实验，不应标注为 LLM 真实结果。

| 系统 | 类型 | 决策准确率 | 三维风险分数 MAE |
|---|---|---:|---:|
| rule_only | offline_proxy | 1.000 | 1.181 |
| rag_augmented_proxy | offline_proxy | 1.000 | 0.708 |
| full_system_proxy | offline_proxy | 1.000 | 0.708 |

消融定义：`rule_only` 只使用触发词；`rag_augmented_proxy` 使用风险类别和维度先验模拟知识库/审核清单增强；`full_system_proxy` 当前与增强代理共享离线决策器，真实多智能体系统需在 API 可用后重新运行。

真实运行入口：`run_deepseek_baselines.py`。
