"""项目配置文件"""
import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# LLM配置
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))

# RAG配置
TOP_K = int(os.getenv("TOP_K", "3"))

# 知识库路径
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MANUALS = {
    "智能手环": os.path.join(DATA_DIR, "smart_band_manual.txt"),
    "智能手表": os.path.join(DATA_DIR, "smart_watch_manual.txt"),
    "智能音箱": os.path.join(DATA_DIR, "smart_speaker_manual.txt"),
    "智能台灯": os.path.join(DATA_DIR, "smart_lamp_manual.txt"),
    "智能门锁": os.path.join(DATA_DIR, "smart_lock_manual.txt"),
}

# 评测配置
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "evaluation", "test_set.csv")

# 拒答信号词(用于识别超纲问题)
REFUSAL_SIGNALS = [
    "不在", "无法", "不能", "没有收录", "不包含", "超出",
    "不提供", "无法作答", "不在手册", "不在指南", "建议咨询"
]
