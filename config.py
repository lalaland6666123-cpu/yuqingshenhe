import os

# ========== DeepSeek V4（文案/文本审核）==========
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "your-deepseek-key")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL_NAME = "deepseek-chat"

# ========== 千问（图片/视频视觉分析）==========
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "your-qwen-key")
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL_NAME = "qwen-plus"
VL_MODEL_NAME = "qwen3.5-plus"

# ========== 向后兼容别名 ==========
MODEL_NAME = DEEPSEEK_MODEL_NAME

# ========== RAG 检索引擎 ==========
EMBEDDING_MODEL_NAME = "text-embedding-v2"
CHROMA_PERSIST_DIR = "data/chroma_db"
RAG_TOP_K = 5

# ========== LLM 温度常量 ==========
TEMP_ANALYSIS = 0.2      # 视觉分析 / JSON 结构化输出
TEMP_SEED_ROSTER = 0.4   # 种子角色生成
TEMP_REPORT = 0.35       # 报告 / 改写建议
TEMP_CATALYST = 0.55     # 催化剂 agent 生成
TEMP_ZEITGEIST = 0.45    # 宏观环境分析
TEMP_SANDBOX_CHAT = 0.9  # 沙盘广场发言（需多样性）
TEMP_DEFAULT = 0.4