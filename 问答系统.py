"""SmartGuide AI - 智能硬件客服助手

单页应用,顶部「用户端 / 服务端」左右切换:
  - 用户端:聊天式问答(输入框在下方,问答气泡在上方)
  - 服务端:管理后台(数据看板 + 知识库管理,需登录)
"""
import streamlit as st
from src.simple_rag import SimpleRAG, DEVICE_KEYWORD_MAP
from src.llm_client import LLMClient
from src.evaluator import Evaluator
from src.admin_panel import render_server
import config


@st.cache_resource
def load_knowledge_base():
    """加载知识库(只加载一次,避免每次刷新重复读文件)"""
    documents = {}
    for device_name, file_path in config.MANUALS.items():
        with open(file_path, 'r', encoding='utf-8') as f:
            documents[device_name] = f.read()
    return SimpleRAG(documents)


@st.cache_resource
def get_llm_client():
    """复用 LLM 客户端(避免每次刷新重复初始化)"""
    return LLMClient()


def extract_device(question: str):
    """从问题里识别设备名(用于多轮上下文记忆)。"""
    for name in config.MANUALS:
        if name in question:
            return name
    for kw, name in DEVICE_KEYWORD_MAP.items():
        if kw in question:
            return name
    return None


def init_session_state():
    """初始化 session_state"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []          # 聊天记录 [{role, content, docs?}]
    if 'history' not in st.session_state:
        st.session_state.history = []           # 指标记录(供服务端)
    if 'context_device' not in st.session_state:
        st.session_state.context_device = None  # 多轮上下文:当前设备
    if 'pending_prompt' not in st.session_state:
        st.session_state.pending_prompt = None  # 待处理的提问(两段式渲染)


def compute_assistant(prompt, rag, llm_client, evaluator):
    """计算回答并追加助手气泡 + 记录指标。"""
    # 多轮上下文:记住用户当前在问哪个设备
    device = extract_device(prompt)
    if device:
        st.session_state.context_device = device

    # 问题没提设备时,沿用上一轮的设备(如"那怎么充电?")
    effective = prompt
    if not device and st.session_state.context_device:
        effective = f"{st.session_state.context_device} {prompt}"

    # 处理流程
    # 注意：先校验用户原话，而不是补全后的 effective。
    # 否则“你好”在已有设备上下文时会被补成“智能手环 你好”，
    # 因包含设备词而绕过无效问题拦截，白白调用一次模型。
    if not evaluator.is_valid_question(prompt):
        answer = "⚠️ 问题无效：不符合客服场景（问题过短、偏离设备使用，或没提到具体动作）。"
        docs, category, accuracy, detail = [], "无效", None, None
    else:
        docs = rag.retrieve(effective, top_k=config.TOP_K)
        if not docs:
            answer = "该问题不在设备使用指南中，无法作答。请换个与设备使用相关的问法。"
            category, accuracy, detail = "超纲", None, None
        else:
            try:
                answer = llm_client.answer_question(effective, docs)
                category, _ = evaluator.classify_question(effective, answer)
                accuracy, detail = (evaluator.calculate_accuracy_a(answer, docs)
                                    if category == "有效" else (None, None))
            except Exception as e:
                answer = f"❌ 调用AI失败：{e}"
                docs, category, accuracy, detail = [], "错误", None, None

    # 记录指标(供服务端数据看板)
    st.session_state.history.append({
        "question": prompt,
        "effective_question": effective,
        "answer": answer,
        "retrieved_docs": docs,
        "category": category,
        "accuracy": accuracy,
        "detail": detail,
        "context_device": st.session_state.context_device,
    })

    # 加入助手气泡
    st.session_state.messages.append({'role': 'assistant', 'content': answer, 'docs': docs})


def render_chat_client(rag, llm_client, evaluator):
    """渲染用户端:聊天式问答界面。"""
    # 多轮上下文提示
    if st.session_state.context_device:
        c1, c2 = st.columns([6, 1])
        c1.caption(f"🧠 上下文记忆：正在聊 **{st.session_state.context_device}**，后续问题可省略设备名")
        if c2.button("清除上下文", use_container_width=True):
            st.session_state.context_device = None
            st.rerun()

    # 欢迎语(无历史时)
    if not st.session_state.messages:
        st.markdown("""
        👋 你好，我是 **SmartGuide AI** 智能硬件客服助手。可以问我：

        - 智能手环如何连接手机？
        - 智能门锁能录入多少个指纹？
        - 智能音箱怎么重置？
        - 智能台灯支持哪些色温？

        *(问过一次设备后，可以省略设备名接着追问)*
        """)

    # 历史消息(问答气泡在上方)
    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])
            # 助手回答附资料出处(折叠,保持界面干净)
            if msg['role'] == 'assistant' and msg.get('docs'):
                with st.expander("📚 资料出处"):
                    for i, (device, content, score) in enumerate(msg['docs'], 1):
                        st.caption(f"[{i}] {device} · 相似度 {score:.3f}")
                        st.text(content[:200] + ("..." if len(content) > 200 else ""))

    # 底部输入框
    prompt = st.chat_input("请输入您的问题，例如：智能手环如何连接手机？")
    if prompt:
        # 先显示用户气泡,再计算回答(两段式渲染,像真实聊天)
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        st.session_state.pending_prompt = prompt
        st.rerun()

    if st.session_state.pending_prompt:
        pending = st.session_state.pending_prompt
        st.session_state.pending_prompt = None
        with st.spinner("🤔 AI思考中..."):
            compute_assistant(pending, rag, llm_client, evaluator)
        st.rerun()


def main():
    st.set_page_config(page_title="SmartGuide AI", page_icon="🤖", layout="wide")

    st.title("🤖 SmartGuide AI - 智能硬件客服助手")
    st.caption("支持的设备：智能手环 · 智能手表 · 智能音箱 · 智能台灯 · 智能门锁")

    init_session_state()
    rag = load_knowledge_base()
    llm_client = get_llm_client()
    evaluator = Evaluator()

    # 顶部「用户端 / 服务端」左右切换
    view = st.segmented_control(
        "视图切换",
        ["🤖 用户端", "🛠️ 服务端"],
        default="🤖 用户端",
        label_visibility="collapsed",
    )

    st.divider()

    if view == "🛠️ 服务端":
        render_server()
    else:
        render_chat_client(rag, llm_client, evaluator)


if __name__ == "__main__":
    main()
