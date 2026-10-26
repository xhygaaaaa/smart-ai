"""SmartGuide AI - 智能硬件客服助手

单页应用,顶部「用户端 / 服务端」左右切换:
  - 用户端:聊天式问答(输入框在下方,问答气泡在上方)
  - 服务端:管理后台(数据看板 + 知识库管理,需登录)
"""
import time
import uuid

import streamlit as st
from src.simple_rag import DEVICE_KEYWORD_MAP
from src.retrieval import build_retriever
from src.llm_client import LLMClient
from src.evaluator import Evaluator
from src.admin_panel import render_server
from src import telemetry
import config


@st.cache_resource
def load_knowledge_base():
    """加载知识库(只加载一次,避免每次刷新重复读文件)"""
    documents = {}
    for device_name, file_path in config.MANUALS.items():
        with open(file_path, 'r', encoding='utf-8') as f:
            documents[device_name] = f.read()
    return build_retriever(documents)


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
    if 'session_id' not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex


def build_history():
    """取当前提问之前的历史对话（不含刚追加的这条用户消息）。

    只保留 role/content，参考文档不进入下文，避免历史文档挤占窗口。
    """
    msgs = st.session_state.messages
    # 最后一条是本轮用户提问
    prior = msgs[:-1] if msgs and msgs[-1].get('role') == 'user' else msgs
    return [{'role': m['role'], 'content': m['content']}
            for m in prior if m.get('role') in ('user', 'assistant') and m.get('content')]


def compute_assistant(prompt, rag, llm_client, evaluator):
    """计算回答并追加助手气泡 + 记录指标。"""
    started = time.time()
    history = build_history()

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
                answer = llm_client.answer_question(
                    effective, docs, history=history,
                    max_turns=config.HISTORY_TURNS,
                )
                category, _ = evaluator.classify_question(effective, answer)
                accuracy, detail = (evaluator.calculate_accuracy_a(answer, docs)
                                    if category == "有效" else (None, None))
            except Exception as e:
                answer = f"❌ 调用AI失败：{e}"
                docs, category, accuracy, detail = [], "错误", None, None

    latency_ms = int((time.time() - started) * 1000)

    # 记录指标(供服务端数据看板)
    record = {
        "question": prompt,
        "effective_question": effective,
        "answer": answer,
        "retrieved_docs": docs,
        "category": category,
        "accuracy": accuracy,
        "detail": detail,
        "context_device": st.session_state.context_device,
        "history_turns": len(history) // 2,
        "latency_ms": latency_ms,
    }
    st.session_state.history.append(record)

    # 持久化埋点（自助解决率看板的数据源）
    event_id = None
    try:
        event_id = telemetry.log_query(
            session_id=st.session_state.session_id,
            question=prompt,
            effective_question=effective,
            answer=answer,
            category=category,
            groundedness=accuracy,
            retrieved_count=len(docs),
            latency_ms=latency_ms,
            device=st.session_state.context_device,
            retriever=config.RETRIEVER,
        )
    except Exception as e:  # 埋点失败不能影响问答主流程
        st.session_state.last_telemetry_error = str(e)

    # 加入助手气泡
    st.session_state.messages.append({
        'role': 'assistant',
        'content': answer,
        'docs': docs,
        'event_id': event_id,
        'feedback': None,
    })


def render_chat_client(rag, llm_client, evaluator):
    """渲染用户端:聊天式问答界面。"""
    # 顶部：当前上下文 + 新对话（清空聊天记录、上下文和会话 ID）
    c1, c2 = st.columns([6, 1])
    with c1:
        if st.session_state.context_device:
            st.caption(f"🧠 上下文记忆：正在聊 **{st.session_state.context_device}**，"
                       "后续问题可省略设备名")
        elif st.session_state.messages:
            st.caption("💬 当前对话进行中")
        else:
            st.caption("💬 新对话")
    if c2.button("🆕 新对话", use_container_width=True,
                 help="清空聊天记录和多轮上下文，重新开始一段对话"):
        st.session_state.messages = []
        st.session_state.context_device = None
        st.session_state.pending_prompt = None
        st.session_state.session_id = uuid.uuid4().hex
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

            # 解决情况反馈：自助解决率的数据来源之一
            if msg['role'] == 'assistant' and msg.get('event_id'):
                if msg.get('feedback') is None:
                    c1, c2, c3 = st.columns([1, 1, 5])
                    if c1.button("👍 已解决", key=f"ok_{msg['event_id']}",
                                 use_container_width=True):
                        telemetry.log_feedback(msg['event_id'], True, "helpful")
                        msg['feedback'] = True
                        st.rerun()
                    if c2.button("👎 未解决", key=f"no_{msg['event_id']}",
                                 use_container_width=True):
                        telemetry.log_feedback(msg['event_id'], False, "unhelpful")
                        msg['feedback'] = False
                        st.rerun()
                else:
                    st.caption("✅ 已记录：已解决" if msg['feedback']
                               else "📝 已记录：未解决，建议转人工")

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
