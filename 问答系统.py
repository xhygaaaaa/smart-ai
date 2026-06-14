"""SmartGuide AI - 主页:问答系统"""
import streamlit as st
from src.simple_rag import SimpleRAG
from src.llm_client import LLMClient
from src.evaluator import Evaluator
import config


@st.cache_resource
def load_knowledge_base():
    """加载知识库(只加载一次)"""
    documents = {}
    for device_name, file_path in config.MANUALS.items():
        with open(file_path, 'r', encoding='utf-8') as f:
            documents[device_name] = f.read()
    return SimpleRAG(documents)


def init_session_state():
    """初始化session_state"""
    if 'current_result' not in st.session_state:
        st.session_state.current_result = None
    if 'history' not in st.session_state:
        st.session_state.history = []


def main():
    st.set_page_config(
        page_title="SmartGuide AI",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 SmartGuide AI - 智能硬件客服助手")
    st.markdown("""
    支持的设备: **智能手环** | **智能手表** | **智能音箱** | **智能台灯** | **智能门锁**

    ---
    """)

    # 初始化
    init_session_state()
    rag = load_knowledge_base()
    llm_client = LLMClient()
    evaluator = Evaluator()

    # 输入区
    col1, col2 = st.columns([4, 1])
    with col1:
        question = st.text_input(
            "请输入您的问题:",
            placeholder="例如: 智能手环如何连接手机?",
            key="question_input"
        )
    with col2:
        st.write("")  # 占位对齐
        submit = st.button("🔍 提问", type="primary", use_container_width=True)

    # 处理提问
    if submit and question:
        # 第一级过滤:无效问题
        if not evaluator.is_valid_question(question):
            st.warning("⚠️ 问题无效:不符合客服场景(问题过短或不包含设备关键词)")
            st.session_state.current_result = None
        else:
            with st.spinner("🔍 检索知识库..."):
                retrieved_docs = rag.retrieve(question, top_k=config.TOP_K)

            if not retrieved_docs:
                st.warning("❌ 未找到相关内容")
                st.session_state.current_result = None
            else:
                with st.spinner("🤔 AI思考中..."):
                    try:
                        answer = llm_client.answer_question(question, retrieved_docs)

                        # 分类和评分
                        category, category_note = evaluator.classify_question(question, answer)

                        # 只对"有效"问题计算准确率
                        if category == "有效":
                            accuracy, detail = evaluator.calculate_accuracy_a(answer, retrieved_docs)
                        else:
                            accuracy, detail = None, None

                        # 存储结果到session_state(解决截图后答案消失的bug)
                        st.session_state.current_result = {
                            "question": question,
                            "answer": answer,
                            "retrieved_docs": retrieved_docs,
                            "category": category,
                            "category_note": category_note,
                            "accuracy": accuracy,
                            "detail": detail
                        }

                        # 记录历史
                        st.session_state.history.append(st.session_state.current_result)

                    except Exception as e:
                        st.error(f"❌ 调用AI失败: {str(e)}")
                        st.session_state.current_result = None

    # 显示结果(从session_state读取,避免rerun后丢失)
    if st.session_state.current_result:
        result = st.session_state.current_result

        st.success("✅ 回答生成完毕")

        # 显示答案
        st.markdown("### 💬 AI回答")
        st.info(result["answer"])

        # 显示分类和准确率
        col1, col2, col3 = st.columns(3)
        with col1:
            category_color = {
                "有效": "🟢",
                "超纲": "🟡",
                "无效": "🔴"
            }
            st.metric("问题分类", f"{category_color[result['category']]} {result['category']}")
        with col2:
            if result["accuracy"] is not None:
                st.metric("方案A准确率", f"{result['accuracy']:.1%}")
            else:
                st.metric("方案A准确率", "不适用")
        with col3:
            st.metric("引用文档数", len(result["retrieved_docs"]))

        # 显示分类说明
        st.caption(f"📝 {result['category_note']}")

        # 显示检索的文档
        with st.expander("📚 查看引用文档"):
            for i, (device, content, score) in enumerate(result["retrieved_docs"], 1):
                st.markdown(f"**[文档{i}] {device}** (相似度: {score:.3f})")
                st.text(content[:200] + "..." if len(content) > 200 else content)
                st.divider()

        # 显示关键词分析(仅有效问题)
        if result["detail"]:
            with st.expander("🔍 关键词匹配详情"):
                detail = result["detail"]
                st.write(f"- 答案关键词总数: {detail['total_keywords']}")
                st.write(f"- 有文档支持的: {detail['supported_keywords']}")
                st.write(f"- 无文档支持的: {len(detail['unsupported_keywords'])}")
                if detail['unsupported_keywords']:
                    st.warning(f"缺乏依据的词: {', '.join(detail['unsupported_keywords'][:10])}")

    # 侧边栏
    with st.sidebar:
        st.header("📊 功能导航")
        st.page_link("问答系统.py", label="🏠 问答系统", icon="🏠")
        st.page_link("pages/实时评测.py", label="📈 实时评测", icon="📈")
        st.page_link("pages/一键评测.py", label="🎯 一键评测", icon="🎯")

        st.divider()

        st.header("ℹ️ 系统信息")
        st.info(f"""
        - **模型**: {config.LLM_MODEL}
        - **检索TOP_K**: {config.TOP_K}
        - **准确率算法**: 方案A(防幻觉检测)
        """)

        if st.session_state.history:
            st.divider()
            st.header("📜 本次会话历史")
            st.caption(f"共 {len(st.session_state.history)} 条提问")


if __name__ == "__main__":
    main()
