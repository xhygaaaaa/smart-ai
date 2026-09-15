"""管理端登录鉴权(服务端各页面共用)

对应「用户端 / 服务端」分离设计:
  - 用户端(问答系统.py)无需登录,只展示干净回答
  - 服务端(服务端.py / 实时评测.py / 一键评测.py)需要管理员密码
"""
import streamlit as st
import config


def require_login() -> bool:
    """校验管理员登录态;未登录时渲染登录框并返回 False。

    登录成功写入 st.session_state.admin_authed,跨页面跳转无需重复登录。
    """
    if st.session_state.get("admin_authed"):
        return True

    st.markdown("### 🔐 管理员登录")
    st.caption("服务端页面仅对管理员开放，请输入密码")

    with st.form("admin_login"):
        pwd = st.text_input("密码", type="password")
        submitted = st.form_submit_button("登录", type="primary", use_container_width=True)

    st.caption("💡 默认密码：admin")

    if submitted:
        if pwd == config.ADMIN_PASSWORD:
            st.session_state.admin_authed = True
            st.rerun()
        else:
            st.error("❌ 密码错误，请重试")

    return False
