"""SmartGuide AI - 一键评测页面(批量跑42题测试集)"""
import streamlit as st
import pandas as pd
import time
import plotly.express as px
from src.simple_rag import SimpleRAG
from src.llm_client import LLMClient
from src.evaluator import Evaluator
import config
from src.auth import require_login


@st.cache_resource
def load_knowledge_base():
    """加载知识库"""
    documents = {}
    for device_name, file_path in config.MANUALS.items():
        with open(file_path, 'r', encoding='utf-8') as f:
            documents[device_name] = f.read()
    return SimpleRAG(documents)


def init_evaluation_state():
    """初始化评测状态"""
    if 'eval_running' not in st.session_state:
        st.session_state.eval_running = False
    if 'eval_results' not in st.session_state:
        st.session_state.eval_results = []
    if 'eval_current_idx' not in st.session_state:
        st.session_state.eval_current_idx = 0
    if 'eval_test_set' not in st.session_state:
        st.session_state.eval_test_set = None


def run_batch_evaluation():
    """分批运行评测(适配Streamlit的rerun机制)"""
    rag = load_knowledge_base()
    llm_client = LLMClient()
    evaluator = Evaluator()

    test_df = st.session_state.eval_test_set
    idx = st.session_state.eval_current_idx

    if idx >= len(test_df):
        # 评测完成
        st.session_state.eval_running = False
        return

    # 处理当前问题
    row = test_df.iloc[idx]
    question = row['question']

    try:
        # 检索
        retrieved_docs = rag.retrieve(question, top_k=config.TOP_K)

        if not retrieved_docs:
            result = {
                "question": question,
                "answer": "未找到相关文档",
                "category": "无效",
                "accuracy": None,
                "error": None
            }
        else:
            # 调用AI(带限流重试)
            answer = llm_client.answer_question(question, retrieved_docs)

            # 分类和评分
            category, _ = evaluator.classify_question(question, answer)

            if category == "有效":
                accuracy, _ = evaluator.calculate_accuracy_a(answer, retrieved_docs)
            else:
                accuracy = None

            result = {
                "question": question,
                "answer": answer,
                "category": category,
                "accuracy": accuracy,
                "error": None
            }

        # 延时避免API限流(如仍频繁429,可改成2或3秒)
        time.sleep(1)

    except Exception as e:
        result = {
            "question": question,
            "answer": "",
            "category": "错误",
            "accuracy": None,
            "error": str(e)
        }

    st.session_state.eval_results.append(result)
    st.session_state.eval_current_idx += 1

    # 触发rerun继续下一题
    st.rerun()


def main():
    st.set_page_config(
        page_title="一键评测 - SmartGuide AI",
        page_icon="🎯",
        layout="wide"
    )

    st.title("🎯 一键评测 - 批量测试42题")
    st.markdown("验证Prompt优化效果，快速跑完整个测试集")

    if not require_login():
        return

    init_evaluation_state()

    # 加载测试集
    try:
        test_df = pd.read_csv(config.TEST_SET_PATH)
        # 过滤掉表头重复行(CSV里第2行是重复表头)
        test_df = test_df[test_df['question'] != 'question']
        test_df = test_df.reset_index(drop=True)
        total_questions = len(test_df)
    except Exception as e:
        st.error(f"❌ 加载测试集失败: {str(e)}")
        return

    # 控制按钮
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        if st.button("▶️ 开始评测", disabled=st.session_state.eval_running, type="primary"):
            st.session_state.eval_running = True
            st.session_state.eval_results = []
            st.session_state.eval_current_idx = 0
            st.session_state.eval_test_set = test_df
            st.rerun()

    with col2:
        if st.button("🔄 重置", disabled=st.session_state.eval_running):
            st.session_state.eval_running = False
            st.session_state.eval_results = []
            st.session_state.eval_current_idx = 0
            st.session_state.eval_test_set = None
            st.rerun()

    # 进度显示
    if st.session_state.eval_running or st.session_state.eval_results:
        current = st.session_state.eval_current_idx
        progress = current / total_questions

        st.progress(progress, text=f"进度: {current}/{total_questions} ({progress:.1%})")

        if st.session_state.eval_running:
            st.info(f"⏳ 正在评测第 {current + 1} 题...")

    # 执行评测
    if st.session_state.eval_running:
        run_batch_evaluation()

    # 显示结果
    if st.session_state.eval_results and not st.session_state.eval_running:
        st.success("✅ 评测完成!")

        results = st.session_state.eval_results

        # 统计
        valid_results = [r for r in results if r["category"] == "有效"]
        valid_accuracies = [r["accuracy"] for r in valid_results if r["accuracy"] is not None]

        st.markdown("### 📊 评测结果")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("总问题数", len(results))
        with col2:
            st.metric("有效问题", len(valid_results))
        with col3:
            if valid_accuracies:
                avg_acc = sum(valid_accuracies) / len(valid_accuracies)
                st.metric("平均准确率", f"{avg_acc:.1%}")
            else:
                st.metric("平均准确率", "无数据")
        with col4:
            error_count = len([r for r in results if r["error"]])
            st.metric("错误数", error_count)

        # 分类统计
        st.markdown("### 📋 问题分类统计")
        categories = [r["category"] for r in results]
        category_counts = pd.Series(categories).value_counts()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("有效", category_counts.get("有效", 0))
        with col2:
            st.metric("超纲", category_counts.get("超纲", 0))
        with col3:
            st.metric("无效/错误", category_counts.get("无效", 0) + category_counts.get("错误", 0))

        # 准确率分布(仅有效问题)
        if valid_accuracies:
            st.markdown("### 📈 准确率分布")

            fig = px.histogram(
                valid_accuracies,
                nbins=10,
                labels={"value": "准确率", "count": "问题数"},
                title="有效问题准确率分布"
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        # 详细结果表
        st.markdown("### 📄 详细结果")

        df_display = pd.DataFrame({
            "序号": range(1, len(results) + 1),
            "问题": [r["question"] for r in results],
            "分类": [r["category"] for r in results],
            "准确率": [f"{r['accuracy']:.1%}" if r["accuracy"] is not None else "-"
                     for r in results],
            "错误": [r["error"] if r["error"] else "" for r in results]
        })

        st.dataframe(df_display, use_container_width=True, hide_index=True)

        # 导出结果
        with st.expander("💾 导出结果(CSV格式)"):
            csv = df_display.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="下载CSV",
                data=csv,
                file_name="evaluation_results.csv",
                mime="text/csv"
            )

    elif not st.session_state.eval_results:
        st.info("👆 点击「开始评测」按钮，批量跑42题测试集")


if __name__ == "__main__":
    main()
