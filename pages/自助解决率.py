"""自助解决率看板

数据源：src/telemetry.py 写入的 SQLite 事件表。
口径：
  已解决 = 用户点了「已解决」
         或 未反馈、且 FOLLOWUP_WINDOW_MINUTES 内没有继续追问
  自助解决率 = 已解决提问数 / 全部提问数
追问（follow_up）视为「没有自助解决」的隐式信号。
"""
import pandas as pd
import plotly.express as px
import streamlit as st

import config
from src.auth import require_login
from src import telemetry


def _with_solved_flag(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    explicit = df["resolved"] == 1
    implicit = df["resolved"].isna() & (df["follow_up"].fillna(0) == 0)
    df = df.copy()
    df["solved"] = explicit | implicit
    df["date"] = pd.to_datetime(df["ts"]).dt.date
    return df


def main():
    st.set_page_config(page_title="自助解决率 - SmartGuide AI", page_icon="🎯", layout="wide")
    st.title("🎯 自助解决率看板")
    st.caption("北极星指标：用户问完没有再追问、也没有转人工的比例")

    if not require_login():
        return

    days = st.selectbox("统计窗口", [1, 7, 30, 90], index=1,
                        format_func=lambda d: f"最近 {d} 天")

    events = telemetry.fetch_events(days)
    summary = telemetry.compute_resolution_rate(events)

    if summary["total"] == 0:
        st.info("窗口期内还没有提问数据。先到「用户端」问几个问题，或点几下反馈按钮。")
        return

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("自助解决率", f"{summary['resolution_rate']:.1%}")
    c2.metric("提问量", summary["total"])
    c3.metric("显式反馈覆盖率", f"{summary['feedback_coverage']:.1%}",
              help="点了「已解决/未解决」的提问占比")
    c4.metric("追问率", f"{summary['follow_up'] / summary['total']:.1%}",
              help=f"同一会话 {config.FOLLOWUP_WINDOW_MINUTES} 分钟内继续追问的比例")
    c5.metric("平均有据可查率",
              f"{summary['avg_groundedness']:.1%}"
              if summary["avg_groundedness"] is not None else "-")

    with st.expander("📐 指标定义与计算方式"):
        st.markdown(f"""
- **已解决**：用户点「👍 已解决」，或未反馈但 {config.FOLLOWUP_WINDOW_MINUTES} 分钟内没有继续追问。
- **未解决**：用户点「👎 未解决」，或未反馈但同一会话内很快追问（`follow_up=1`）。
- **自助解决率** = 已解决提问 / 全部提问。
- **局限**：未反馈且没追问 ≠ 一定解决（用户可能直接离开）。因此同时展示显式反馈覆盖率；
  显式反馈越多，指标越可信。
- 数据表：`data/telemetry.db`，事件级可追溯到问题、回答、检索数和耗时。
        """)

    df = _with_solved_flag(pd.DataFrame(events))

    st.markdown("### 📈 趋势")
    daily = (df.groupby("date")
               .agg(提问量=("id", "count"), 已解决=("solved", "sum"))
               .reset_index())
    daily["自助解决率"] = daily["已解决"] / daily["提问量"]
    fig = px.line(daily, x="date", y="自助解决率", markers=True,
                  title="每日自助解决率")
    fig.update_yaxes(tickformat=".0%", range=[0, 1.05])
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🧩 问题分类分布")
        counts = df["category"].fillna("未知").value_counts().reset_index()
        counts.columns = ["分类", "数量"]
        st.plotly_chart(
            px.pie(counts, names="分类", values="数量", hole=0.4),
            use_container_width=True,
        )
    with col2:
        st.markdown("### ❓ 未解决信号")
        unsolved = df[(df["resolved"] == 0) | (df["follow_up"] == 1)]
        st.metric("未解决/追问提问数", len(unsolved))
        if not unsolved.empty:
            top = (unsolved["question"].value_counts().head(8)
                   .reset_index())
            top.columns = ["问题", "次数"]
            st.dataframe(top, use_container_width=True, hide_index=True)
        else:
            st.caption("暂无未解决信号")

    st.markdown("### 📄 事件明细")
    show = df[["ts", "question", "category", "groundedness",
               "retrieved_count", "latency_ms", "resolved", "follow_up", "solved"]]
    st.dataframe(show, use_container_width=True, hide_index=True)
    st.download_button(
        "下载事件明细 CSV",
        data=show.to_csv(index=False).encode("utf-8-sig"),
        file_name="telemetry_events.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
