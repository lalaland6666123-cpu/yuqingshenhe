# PR Wind Tunnel — 智能文案审查系统

项目路径: `D:\ai daima\PR_Wind_Tunnel`

## 这是什么
基于多智能体 LLM 的公关危机舆情沙盘推演系统。
用户输入事件描述和公关草稿 → RAG 检索匹配危机模式 → Master Agent 生成角色阵容 → 4 轮模拟舆论对线 → 生成风险评估报告。

## 开发约定
- 所有文本生成走 DeepSeek (`deepseek-chat`)，多模态分析走 Qwen (`qwen3.5-plus`)
- RAG 使用 ChromaDB + DashScope Embedding，知识库 JSON 在 `data/` 目录
- Streamlit 前端，宽屏布局，办公平台浅色风格
- 温度参数集中在 `config.py` 管理，不要硬编码 magic number
- API Key 在 `config.py` 中，注意不要提交到公开仓库

## RAG 知识体系 (v2.0)
当前采用「分类框架 + 关键词模板 + 审查清单」三层结构：
1. `data/crisis_taxonomy.json` — 8 大类 40+ 子类危机模式，每个含关键词触发热词和思考引导
2. `data/keyword_templates.json` — 场景模板，用关键词组合替代完整案例
3. `data/audit_checklist.json` — 4 阶段审查清单，按 Agent 角色分配

## 核心文件
- `app.py` — 主应用 (~1700 行)
- `rag_engine.py` — RAG v2.0 检索引擎
- `evaluator.py` — 确定性评估 (Legal/Business/Reputation)
- `spread_model.py` — SIR 传播动力学
- `config.py` — API Key + 模型 + 温度配置
- `build_db.py` — 知识库建设 + ChromaDB 索引构建

## 记忆位置
项目记忆文件在 `C:\Users\裴\.claude\projects\D--ai-daima-PR-Wind-Tunnel\memory\`
包含: 项目总览、RAG 设计决策、待执行改进计划、API 配置
