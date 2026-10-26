import pytest

import config
from src.retrieval import BM25RAG, build_retriever, tokenize
from src.simple_rag import SimpleRAG

DOCS = {
    "智能手环": (
        "电池续航\n"
        "智能手环充满电后可使用 7 天，支持 5ATM 防水。\n\n"
        "连接手机\n"
        "打开手机蓝牙，在 App 内添加设备，搜索到手环后完成配对。\n\n"
        "充电方法\n"
        "使用原装充电器连接充电触点，约 2 小时充满。"
    ),
    "智能门锁": (
        "指纹录入\n"
        "智能门锁最多可录入 100 枚指纹。\n\n"
        "电池续航\n"
        "门锁电池可使用 12 个月。"
    ),
}


def test_tokenize_mixed():
    toks = tokenize("支持5ATM防水，续航7天")
    assert "5atm" in toks
    assert "防水" in toks


def test_bm25_device_routing():
    rag = BM25RAG(DOCS)
    hits = rag.retrieve("智能门锁能录几个指纹？", top_k=5)
    assert hits
    assert all(device == "智能门锁" for device, _, _ in hits)
    assert "100" in hits[0][1]


def test_bm25_returns_normalized_scores():
    rag = BM25RAG(DOCS)
    hits = rag.retrieve("手环续航多久", top_k=3)
    assert hits[0][2] == 1.0
    assert all(0.0 <= score <= 1.0 for _, _, score in hits)


def test_build_retriever_modes(monkeypatch):
    assert isinstance(build_retriever(DOCS, "bigram"), SimpleRAG)
    assert isinstance(build_retriever(DOCS, "bm25"), BM25RAG)
    # 模拟向量依赖不可用：hybrid 应自动降级为 bm25（不触发模型下载）
    monkeypatch.setattr("src.retrieval._try_dense", lambda *a, **k: None)
    assert isinstance(build_retriever(DOCS, "hybrid"), BM25RAG)


def test_bigram_unchanged():
    rag = SimpleRAG(DOCS)
    hits = rag.retrieve("智能手环怎么连接手机", top_k=2)
    assert hits and all(d == "智能手环" for d, _, _ in hits)


def load_real_manuals():
    docs = {}
    for device, path in config.MANUALS.items():
        with open(path, "r", encoding="utf-8") as f:
            docs[device] = f.read()
    return docs


@pytest.mark.parametrize("question", ["智能音箱怎么重置？", "这个智能音箱怎么重置？"])
@pytest.mark.parametrize("retriever", [SimpleRAG, BM25RAG])
def test_reset_question_hits_reset_section(retriever, question):
    """欢迎页的推荐问题必须能检索到答案。

    “重置”是用户口语，手册里写的是“恢复出厂设置”，两边字面几乎不重合，
    靠 QUERY_SYNONYMS 做同义桥接才召得回来。这里用真实知识库防止回退。
    """
    rag = retriever(load_real_manuals())
    hits = rag.retrieve(question, top_k=3)
    assert hits, "检索结果为空"
    device, chunk, _ = hits[0]
    assert device == "智能音箱"
    assert "恢复出厂设置" in chunk or "配网键" in chunk
