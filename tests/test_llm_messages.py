import config
from src.llm_client import build_messages

DOCS = [("智能手环", "长按电源键 3 秒开机。", 0.9)]


def test_history_included_in_messages(monkeypatch):
    monkeypatch.setattr(config, "HISTORY_TURNS", 3)
    history = [
        {"role": "user", "content": "智能手环怎么开机？"},
        {"role": "assistant", "content": "长按电源键 3 秒。"},
    ]
    messages = build_messages("那怎么充电？", DOCS, history=history)
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user", "assistant", "user"]
    assert "长按电源键" in messages[2]["content"]
    assert "参考文档" in messages[-1]["content"]


def test_history_window_truncates(monkeypatch):
    monkeypatch.setattr(config, "HISTORY_TURNS", 1)
    history = [
        {"role": "user", "content": "第一轮问题"},
        {"role": "assistant", "content": "第一轮回答"},
        {"role": "user", "content": "第二轮问题"},
        {"role": "assistant", "content": "第二轮回答"},
    ]
    messages = build_messages("当前问题", DOCS, history=history)
    contents = " ".join(m["content"] for m in messages)
    assert "第一轮问题" not in contents
    assert "第二轮问题" in contents


def test_history_can_be_disabled(monkeypatch):
    monkeypatch.setattr(config, "HISTORY_TURNS", 0)
    history = [{"role": "user", "content": "旧问题"},
               {"role": "assistant", "content": "旧回答"}]
    messages = build_messages("新问题", DOCS, history=history)
    assert [m["role"] for m in messages] == ["system", "user"]
