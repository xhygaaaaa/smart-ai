"""SmartGuide AI - 一键评测页面（批量跑标准测试集）

评测在**独立子进程**里跑（`evaluation/bg_runner.py`），进度实时落盘：
切到别的页面、刷新浏览器、关标签页都不会中断，回来还能接着看。

输出两类指标，避免把「有据可查」当成「答得对」：
  - groundedness：答案关键词有多少能在检索文档中找到支撑（防幻觉）
  - correctness ：答案覆盖标准答案关键点的召回率（是否答对）
运行结束后逐题结果自动写入 evaluation/results/<run_id>/。
"""
import json
import os
import subprocess
import sys
import time

import pandas as pd
import plotly.express as px
import streamlit as st

import config
from src.auth import require_login
from src.evaluation_runner import load_test_set

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POINTER = os.path.join(config.EVAL_RESULT_DIR, ".current_run")


def run_dir(run_id: str) -> str:
    return os.path.join(config.EVAL_RESULT_DIR, run_id)


def read_progress(run_id: str):
    path = os.path.join(run_dir(run_id), "progress.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def load_results(run_id: str):
    """评测结束后从 results.jsonl 读逐题结果；跑的过程中用 progress 里的快照。"""
    path = os.path.join(run_dir(run_id), "results.jsonl")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def discover_run_id():
    """优先 session_state，其次指针文件，最后找最新的运行目录。"""
    if st.session_state.get("eval_run_id"):
        return st.session_state.eval_run_id
    if os.path.exists(POINTER):
        try:
            with open(POINTER, "r", encoding="utf-8") as f:
                rid = f.read().strip()
            if rid and os.path.isdir(run_dir(rid)):
                return rid
        except OSError:
            pass
    if not os.path.isdir(config.EVAL_RESULT_DIR):
        return None
    candidates = [
        d for d in os.listdir(config.EVAL_RESULT_DIR)
        if d.startswith("run_") and os.path.exists(
            os.path.join(config.EVAL_RESULT_DIR, d, "progress.json"))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda d: os.path.getmtime(
        os.path.join(config.EVAL_RESULT_DIR, d, "progress.json")))


def start_run(judge: str, retriever: str, sleep_s: float) -> str:
    run_id = time.strftime("run_%Y%m%d_%H%M%S")
    out = run_dir(run_id)
    os.makedirs(out, exist_ok=True)
    log_path = os.path.join(out, "eval.log")
    cmd = [
        sys.executable, "-m", "evaluation.bg_runner",
        "--judge", judge, "--retriever", retriever,
        "--sleep", str(sleep_s), "--run-id", run_id,
    ]
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    with open(log_path, "ab") as log:
        subprocess.Popen(
            cmd, cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, **kwargs,
        )
    with open(POINTER, "w", encoding="utf-8") as f:
        f.write(run_id)
    st.session_state.eval_run_id = run_id
    return run_id


def stop_run(run_id: str):
    """用停止标记文件通知子进程收尾，跨平台且不依赖进程句柄。"""
    with open(os.path.join(run_dir(run_id), "STOP"), "w", encoding="utf-8") as f:
        f.write("stop")


def render_results(results, summary, out_dir):
    st.success("✅ 评测完成，结果已写入磁盘")

    st.markdown("### 📊 核心指标")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("总问题数", summary["total"])
    c2.metric("有效问题", summary["valid"])
    c3.metric("有据可查率", f"{summary['groundedness_avg']:.1%}"
              if summary["groundedness_avg"] is not None else "-")
    c4.metric("关键点召回率", f"{summary['correctness_recall_avg']:.1%}"
              if summary["correctness_recall_avg"] is not None else "-")
    c5.metric("任务成功率", f"{summary['task_success_rate']:.1%}"
              if summary["task_success_rate"] is not None else "-")

    st.caption(
        "有据可查率 = 答案关键词被检索文档覆盖的比例（groundedness）；"
        "关键点召回率 = 答案覆盖标准答案关键词的比例；"
        "任务成功率 = 在可判分题目上，答对 + 超纲题正确拒答的比例。"
    )

    if out_dir:
        st.code(out_dir, language=None)

    df = pd.DataFrame(results)
    st.markdown("### 📋 分类统计")
    if not df.empty and "gold_category" in df:
        counts = df.groupby(["gold_category", "category"]).size().reset_index(name="count")
        st.dataframe(counts, use_container_width=True, hide_index=True)

    if not df.empty and df["groundedness"].notna().any():
        st.markdown("### 📈 有据可查率分布")
        fig = px.histogram(
            df[df["groundedness"].notna()], x="groundedness", nbins=10,
            labels={"groundedness": "有据可查率", "count": "题目数"},
        )
        fig.update_layout(showlegend=False, yaxis_title="题目数")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📄 逐题明细")
    display = pd.DataFrame({
        "问题": df["question"],
        "标准分类": df["gold_category"],
        "判定": df["category"],
        "有据可查": df["groundedness"].map(
            lambda x: f"{x:.1%}" if pd.notna(x) else "-"),
        "关键点召回": df["correctness_recall"].map(
            lambda x: f"{x:.1%}" if pd.notna(x) else "-"),
        "答对": df["correct"].map(lambda x: "✅" if x is True else ("❌" if x is False else "-")),
        "拒答": df["refused"].map(lambda x: "是" if x is True else "-"),
        "耗时(ms)": df["latency_ms"],
        "错误": df["error"].fillna(""),
    })
    st.dataframe(display, use_container_width=True, hide_index=True)

    with st.expander("💾 导出逐题结果(CSV)"):
        st.download_button(
            label="下载CSV",
            data=display.to_csv(index=False).encode("utf-8-sig"),
            file_name="evaluation_results.csv",
            mime="text/csv",
        )

    wrong = df[(df["correct"] == False) | (df["error"].notna())]  # noqa: E712
    if not wrong.empty:
        st.markdown("### 🔍 未通过题目")
        for _, row in wrong.iterrows():
            with st.expander(f"{row['question']}（{row['category']}）"):
                st.markdown(f"**标准答案**：{row['expected_answer']}")
                st.markdown(f"**模型回答**：{row['answer']}")
                if row.get("llm_judge"):
                    st.caption(f"LLM judge: {row['llm_judge']}")


@st.fragment(run_every=2)
def render_running(run_id: str):
    """运行中：只刷新这一块，切页面也不影响后台进程。"""
    state = read_progress(run_id)
    if not state:
        st.info("已提交评测，等待后台进程启动…")
        return
    done, total = state.get("done", 0), state.get("total", 0) or 1
    status = state.get("status")
    if status == "running":
        st.progress(min(done / total, 1.0),
                    text=f"{done}/{total} · {state.get('current') or '准备中...'}")
        c1, c2, c3 = st.columns([1, 1, 4])
        c1.caption(f"run `{run_id}`")
        c2.caption(f"更新于 {state.get('updated_at', '-')}")
        if c3.button("⏹ 停止评测", key=f"stop_{run_id}"):
            stop_run(run_id)
            st.toast("已发送停止请求，当前题跑完后收尾")
        st.caption("可以切到别的页面，评测在后台继续；回到本页会自动显示最新进度。")
        return

    # 终态：直接整页刷新，把结果渲染出来
    st.rerun()


def main():
    st.set_page_config(page_title="一键评测 - SmartGuide AI", page_icon="🎯", layout="wide")
    st.title("🎯 一键评测 - 批量测试标准集")
    st.markdown("可复现评测：逐题落盘 + groundedness / correctness 双指标")

    if not require_login():
        return

    try:
        rows = load_test_set()
    except Exception as e:
        st.error(f"❌ 加载测试集失败: {e}")
        return

    run_id = discover_run_id()
    state = read_progress(run_id) if run_id else None
    running = bool(state and state.get("status") == "running")

    c1, c2, c3 = st.columns([1, 1, 2])
    judge = c1.selectbox("判分方式", ["keyword", "llm"],
                         help="keyword=关键词召回；llm=调用模型判分（更准但更慢）")
    retriever = c2.selectbox("检索后端", ["bm25", "bigram", "hybrid"],
                             index=["bm25", "bigram", "hybrid"].index(config.RETRIEVER)
                             if config.RETRIEVER in ("bm25", "bigram", "hybrid") else 0)
    sleep_s = c3.slider("每题间隔(秒)", 0.0, 3.0, 1.0, 0.5,
                        help="避免触发 API 限流")

    if st.button("▶️ 开始评测", type="primary",
                 disabled=len(rows) == 0 or running,
                 help="已有评测在跑，先停止或等它结束" if running else None):
        rid = start_run(judge, retriever, sleep_s)
        st.session_state.eval_run_id = rid
        st.rerun()

    if not run_id:
        st.info(f"👆 选择参数后点击「开始评测」，批量跑 {len(rows)} 题测试集"
                "（后台运行，切页面不中断）")
        return

    state = read_progress(run_id)
    if not state:
        st.info("已提交评测，等待后台进程启动…")
        render_running(run_id)
        return

    status = state.get("status")
    if status == "running":
        render_running(run_id)
        return

    if status == "stopped":
        st.warning(f"⏹ 已停止（{state.get('done')}/{state.get('total')} 题），"
                   "已跑完的题目结果仍可查看")
    elif status == "failed":
        st.error(f"❌ 评测失败: {state.get('error')}")
        with st.expander("查看 traceback"):
            st.code(state.get("traceback", ""), language="text")
        log_path = os.path.join(run_dir(run_id), "eval.log")
        if os.path.exists(log_path):
            with st.expander("查看后台日志"):
                st.code(open(log_path, encoding="utf-8", errors="replace").read()[-4000:],
                        language="text")

    summary = state.get("summary")
    results = load_results(run_id) or state.get("results") or []
    if summary and results:
        if status == "stopped":
            st.caption("以下为已跑完部分的汇总")
        render_results(results, summary, run_dir(run_id))
    elif results:
        st.caption(f"已有 {len(results)} 条结果，但未生成汇总")
    else:
        st.info("这次没有产生结果")


if __name__ == "__main__":
    main()
