"""项目配置文件"""
import os
from dotenv import load_dotenv

# 加载 .env 文件(本地开发)
load_dotenv()


def _get(key: str, default: str = ""):
    """读取配置:本地读 .env 环境变量;Streamlit Cloud 退而读 secrets.toml。"""
    val = os.getenv(key)
    if val is not None and val != "":
        return val
    try:
        import streamlit as st
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return default


# LLM配置
LLM_BASE_URL = _get("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = _get("LLM_API_KEY", "")
LLM_MODEL = _get("LLM_MODEL", "gpt-3.5-turbo")
LLM_TEMPERATURE = float(_get("LLM_TEMPERATURE", "0.3"))

# RAG配置
TOP_K = int(_get("TOP_K", "3"))

# 管理端密码(服务端页面登录用,可在 .env / secrets.toml 中覆盖)
ADMIN_PASSWORD = _get("ADMIN_PASSWORD", "admin")

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
# 注意：只保留"完整拒答短语"，不能放"无法/不能/不在"这类单字词——
# 否则"手环无法连接蓝牙"这种正常排查答案会被误判成超纲。
REFUSAL_SIGNALS = [
    "不在设备使用指南", "不在使用指南", "无法作答", "无法回答",
    "无法提供", "没有收录", "不包含", "不提供", "不在手册", "不在指南",
    "建议咨询", "超出知识库", "信息不足"
]
