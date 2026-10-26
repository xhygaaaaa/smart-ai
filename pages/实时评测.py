"""SmartGuide AI - 实时评测页面"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.auth import require_login


def init_session_state():
    """初始化session_state"""
    if 'history' not in st.session_state:
        st.session_state.history = []


def main():
    st.set_page_config(
        page_title="实时评测 - SmartGuide AI",
        page_icon="📈",
        layout="wide"
    )

    st.title("📈 实时评测 - 有据可查率统计")
    st.markdown("实时展示用户提问的有据可查率（groundedness）变化趋势")

    if not require_login():
        return

    init_session_state()

    if not st.session_state.history:
        st.warning("⚠️ 暂无提问记录，请先在主页进行提问")
        return

    # 筛选有效问题(只统计有效问题的有据可查率)
    valid_records = [
        r for r in st.session_state.history
        if r.get("category") == "有效" and r.get("accuracy") is not None
    ]

    if not valid_records:
        st.info("📊 当前所有问题均为超纲或无效，无有据可查率数据")

        # 显示分类统计
        categories = [r.get("category", "未知") for r in st.session_state.history]
        category_counts = pd.Series(categories).value_counts()

        st.markdown("### 📋 问题分类统计")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("有效问题", category_counts.get("有效", 0))
        with col2:
            st.metric("超纲问题", category_counts.get("超纲", 0))
        with col3:
            st.metric("无效问题", category_counts.get("无效", 0))
        return

    # 构建数据
    questions = [r["question"] for r in valid_records]
    accuracies = [r["accuracy"] for r in valid_records]
    avg_accuracy = sum(accuracies) / len(accuracies)

    # 显示总体统计
    st.markdown("### 📊 总体统计")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("有效问题数", len(valid_records))
    with col2:
        st.metric("平均有据可查率", f"{avg_accuracy:.1%}")
    with col3:
        st.metric("最高有据可查率", f"{max(accuracies):.1%}")
    with col4:
        st.metric("最低有据可查率", f"{min(accuracies):.1%}")

    # 可视化
    st.markdown("### 📉 有据可查率变化趋势")

    # 创建子图(柱状图+折线图)
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("每次提问有据可查率 (柱状图)", "有据可查率趋势 (折线图)"),
        vertical_spacing=0.15,
        row_heights=[0.5, 0.5]
    )

    # 柱状图
    colors = ['#1f77b4' if acc >= 0.7 else '#ff7f0e' if acc >= 0.5 else '#d62728'
              for acc in accuracies]

    fig.add_trace(
        go.Bar(
            x=list(range(1, len(accuracies) + 1)),
            y=accuracies,
            marker_color=colors,
            text=[f"{acc:.1%}" for acc in accuracies],
            textposition='outside',
            name="有据可查率",
            hovertemplate="<b>第%{x}次提问</b><br>有据可查率: %{y:.1%}<extra></extra>"
        ),
        row=1, col=1
    )

    # 折线图(带滑动平均)
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(accuracies) + 1)),
            y=accuracies,
            mode='lines+markers',
            name="有据可查率",
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=8),
            hovertemplate="<b>第%{x}次</b><br>有据可查率: %{y:.1%}<extra></extra>"
        ),
        row=2, col=1
    )

    # 添加平均线
    fig.add_hline(
        y=avg_accuracy,
        line_dash="dash",
        line_color="red",
        annotation_text=f"平均: {avg_accuracy:.1%}",
        annotation_position="right",
        row=2, col=1
    )

    fig.update_xaxes(title_text="提问序号", row=1, col=1)
    fig.update_xaxes(title_text="提问序号", row=2, col=1)
    fig.update_yaxes(title_text="有据可查率", range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(title_text="有据可查率", range=[0, 1.05], row=2, col=1)

    fig.update_layout(height=800, showlegend=False)

    st.plotly_chart(fig, use_container_width=True)

    # 详细记录表
    st.markdown("### 📋 详细记录")
    df = pd.DataFrame({
        "序号": range(1, len(valid_records) + 1),
        "问题": questions,
        "有据可查率": [f"{acc:.1%}" for acc in accuracies],
        "分类": ["有效"] * len(valid_records)
    })

    st.dataframe(df, use_container_width=True, hide_index=True)

    # 显示完整统计(包含超纲和无效)
    with st.expander("🔍 查看完整统计(含超纲/无效问题)"):
        all_categories = [r.get("category", "未知") for r in st.session_state.history]
        category_counts = pd.Series(all_categories).value_counts()

        st.write("**问题分类分布:**")
        for cat, count in category_counts.items():
            st.write(f"- {cat}: {count} 条")

        st.caption("📝 说明: 只有'有效'分类的问题才会计入有据可查率统计")


if __name__ == "__main__":
    main()
