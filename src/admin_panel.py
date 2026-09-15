"""服务端(管理后台)视图 — 作为主页的一个 tab 渲染

对应「用户端 / 服务端」分离设计:
  - 数据看板:汇总会话提问指标
  - 知识库管理:新增/删除手册内容
"""
import os
import re
import streamlit as st
import pandas as pd
from src.auth import require_login
import config


def load_manual(device_name: str) -> str:
    """读取某设备的手册原文。"""
    path = config.MANUALS[device_name]
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""


def save_manual(device_name: str, content: str):
    """写回某设备的手册。"""
    with open(config.MANUALS[device_name], 'w', encoding='utf-8') as f:
        f.write(content)


def split_paragraphs(content: str):
    """与 SimpleRAG 一致,按空行切分段落。"""
    return [b.strip() for b in re.split(r'\n\s*\n', content)
            if b.strip() and len(b.strip()) > 10]


def metrics_dashboard():
    """数据看板:展示本次会话提问指标。"""
    st.subheader("📊 数据看板")

    history = st.session_state.get("history", [])
    if not history:
        st.info("暂无提问记录。请先到「用户端」提问，指标会汇总到这里。")
    else:
        valid = [r for r in history if r.get("category") == "有效"]
        accs = [r["accuracy"] for r in valid if r.get("accuracy") is not None]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("总提问", len(history))
        c2.metric("有效问题", len(valid))
        c3.metric("超纲问题", len([r for r in history if r.get("category") == "超纲"]))
        c4.metric("平均准确率(方案A)", f"{sum(accs)/len(accs):.1%}" if accs else "无")

        st.markdown("**提问明细**")
        df = pd.DataFrame({
            "问题": [r["question"] for r in history],
            "分类": [r.get("category", "") for r in history],
            "方案A准确率": [f"{r['accuracy']:.1%}" if r.get("accuracy") is not None else "-"
                           for r in history],
            "引用文档数": [len(r.get("retrieved_docs", [])) for r in history],
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("**完整评测工具**")
    st.page_link("pages/实时评测.py", label="📈 实时评测(准确率趋势图)", icon="📈")
    st.page_link("pages/一键评测.py", label="🎯 一键评测(42题批量回归)", icon="🎯")


def knowledge_management():
    """知识库管理:新增/删除手册内容。"""
    st.subheader("📚 知识库管理")

    device = st.selectbox("选择设备手册", list(config.MANUALS.keys()))
    current = load_manual(device)
    paragraphs = split_paragraphs(current)

    st.caption(f"《{device}》当前共 {len(paragraphs)} 个段落")

    with st.expander("👁️ 查看当前手册内容"):
        for i, p in enumerate(paragraphs, 1):
            st.markdown(f"**[{i}]** {p[:120]}{'...' if len(p) > 120 else ''}")
            st.divider()

    # 新增
    st.markdown("#### ➕ 新增内容")
    with st.form("kb_add"):
        new_content = st.text_area(
            "粘贴要新增的内容(会自动追加到该手册末尾,建议是一个完整段落)",
            height=150,
        )
        add_submitted = st.form_submit_button("追加到知识库", type="primary")
    if add_submitted:
        if not new_content.strip():
            st.warning("内容不能为空")
        else:
            sep = "\n\n" if current and not current.endswith("\n") else "\n"
            to_append = sep + new_content.strip() + "\n"
            save_manual(device, current + to_append)
            st.cache_resource.clear()  # 让用户端重新加载知识库
            st.success(f"✅ 已追加到《{device}》，知识库已刷新")
            st.rerun()

    # 删除
    st.markdown("#### ➖ 删除段落")
    if not paragraphs:
        st.caption("当前无段落可删除")
    else:
        labels = [f"[{i}] {p[:40]}..." for i, p in enumerate(paragraphs, 1)]
        to_delete = st.multiselect("勾选要删除的段落(可多选)", labels)
        if st.button("删除所选段落"):
            if not to_delete:
                st.warning("请先勾选要删除的段落")
            else:
                idxs = {int(l.split("]")[0].strip("[")) - 1 for l in to_delete}
                kept = [p for i, p in enumerate(paragraphs) if i not in idxs]
                save_manual(device, "\n\n".join(kept) + ("\n" if kept else ""))
                st.cache_resource.clear()
                st.success(f"✅ 已删除 {len(idxs)} 个段落，知识库已刷新")
                st.rerun()


def render_server():
    """渲染服务端视图(带登录门)。"""
    if not require_login():
        return

    c1, c2 = st.columns([6, 1])
    c1.markdown("✅ 已登录：管理员")
    if c2.button("🚪 退出登录", use_container_width=True):
        st.session_state.admin_authed = False
        st.rerun()

    st.divider()
    metrics_dashboard()

    st.divider()
    knowledge_management()
