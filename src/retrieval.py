"""可切换的检索层：bigram / BM25 / 向量混合（可选 rerank）

设计目标：
  - 默认走 BM25：不依赖向量库，长文档上比字面 bigram 更稳；
  - 规模上来后可切 hybrid：BM25 负责精确实体（型号、数字），
    向量负责口语化表达，再用 RRF 融合；
  - 依赖缺失时自动降级，不阻塞主流程（sentence-transformers 不是必装项）。

对外接口与 SimpleRAG 保持一致：
    retriever.retrieve(query, top_k) -> [(设备名, 段落, 分数), ...]
"""
import math
import re
from typing import Dict, List, Optional, Tuple

import config
from src.simple_rag import DEVICE_KEYWORD_MAP, SimpleRAG

Chunk = Tuple[str, str]  # (设备名, 段落)


def split_chunks(content: str) -> List[str]:
    """按空行切分段落，与 SimpleRAG 保持一致。"""
    return [block.strip() for block in re.split(r'\n\s*\n', content)
            if block.strip() and len(block.strip()) > 10]


def tokenize(text: str) -> List[str]:
    """中英混合分词：中文 bigram + 英文/数字整词。

    不引入 jieba 这类词典分词，是为了保持零依赖；
    BM25 对这种 token 的区分度已经足够。
    """
    text = text.lower()
    tokens: List[str] = []

    for run in re.findall(r'[一-龥]+', text):
        if len(run) == 1:
            tokens.append(run)
        for i in range(len(run) - 1):
            tokens.append(run[i:i + 2])

    tokens.extend(re.findall(r'[a-z0-9]+', text))
    return tokens


def detect_device(query: str) -> Tuple[Optional[str], Optional[str]]:
    """识别问题指向的设备，返回 (设备全名, 命中的设备词)。"""
    for kw, name in DEVICE_KEYWORD_MAP.items():
        if kw in query:
            return name, kw
    return None, None


def expand_query(query: str) -> str:
    """复用 SimpleRAG 的口语->术语同义扩展。"""
    expanded = query
    for kw, extra in SimpleRAG.QUERY_SYNONYMS.items():
        if kw in query:
            expanded += " " + extra
    return expanded


class BM25RAG:
    """BM25 检索器（带设备定向与小节标题加权）"""

    def __init__(self, documents: Dict[str, str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[Chunk] = []
        self.tokens: List[List[str]] = []
        self.doc_freq: Dict[str, int] = {}
        self.avgdl = 0.0

        for device, content in documents.items():
            for chunk in split_chunks(content):
                toks = self._chunk_tokens(chunk)
                self.chunks.append((device, chunk))
                self.tokens.append(toks)
                for t in set(toks):
                    self.doc_freq[t] = self.doc_freq.get(t, 0) + 1

        n = len(self.tokens)
        self.avgdl = sum(len(t) for t in self.tokens) / n if n else 0.0

    @staticmethod
    def _chunk_tokens(chunk: str) -> List[str]:
        """正文 token + 小节标题 token 重复一次（标题命中加权）。"""
        heading, _, body = chunk.partition("\n")
        return tokenize(body) + tokenize(heading) * 2

    def _idf(self, token: str) -> float:
        n = len(self.tokens)
        df = self.doc_freq.get(token, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
        target_device, device_word = detect_device(query)
        term_query = query.replace(device_word, "") if device_word else query
        query_tokens = tokenize(expand_query(term_query))
        if not query_tokens:
            return []

        scored: List[Tuple[str, str, float]] = []
        for (device, chunk), toks in zip(self.chunks, self.tokens):
            if target_device and device != target_device:
                continue

            dl = len(toks) or 1
            freq: Dict[str, int] = {}
            for t in toks:
                freq[t] = freq.get(t, 0) + 1

            score = 0.0
            for t in query_tokens:
                tf = freq.get(t, 0)
                if not tf:
                    continue
                score += self._idf(t) * (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                )

            if score > 0:
                scored.append((device, chunk, score))

        scored.sort(key=lambda x: x[2], reverse=True)
        top = scored[:top_k]
        # 分数归一到 0~1，便于前端展示与跨检索器比较
        if top:
            best = top[0][2] or 1.0
            top = [(d, c, s / best) for d, c, s in top]
        return top


class DenseRAG:
    """向量检索（sentence-transformers，可选依赖）。

    模型不可用时构造返回 None，由 build_retriever 降级处理。
    """

    def __init__(self, documents: Dict[str, str], model_name: Optional[str] = None,
                 chunk_cache: Optional[List[Chunk]] = None):
        from sentence_transformers import SentenceTransformer  # 延迟导入

        self.model = SentenceTransformer(model_name or config.EMBEDDING_MODEL)
        self.chunks: List[Chunk] = chunk_cache or [
            (device, chunk)
            for device, content in documents.items()
            for chunk in split_chunks(content)
        ]
        self.embeddings = self.model.encode(
            [c for _, c in self.chunks], normalize_embeddings=True,
            show_progress_bar=False,
        )

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
        target_device, device_word = detect_device(query)
        term_query = query.replace(device_word, "") if device_word else query
        q = self.model.encode([expand_query(term_query)], normalize_embeddings=True)[0]

        scored = []
        for (device, chunk), emb in zip(self.chunks, self.embeddings):
            if target_device and device != target_device:
                continue
            sim = float(sum(a * b for a, b in zip(q, emb)))
            scored.append((device, chunk, sim))

        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:top_k]


class HybridRAG:
    """BM25 + 向量检索，RRF 融合；可选 CrossEncoder 重排。"""

    def __init__(self, bm25: BM25RAG, dense: DenseRAG,
                 alpha: float = 0.5, reranker=None):
        self.bm25 = bm25
        self.dense = dense
        self.alpha = alpha
        self.reranker = reranker

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
        # 先各召回一批，再融合
        pool = max(top_k * 4, 20)
        bm25_hits = self.bm25.retrieve(query, top_k=pool)
        dense_hits = self.dense.retrieve(query, top_k=pool)

        # Reciprocal Rank Fusion
        rrf: Dict[Tuple[str, str], float] = {}
        chunk_map: Dict[Tuple[str, str], str] = {}
        k = 60
        for weight, hits in ((1 - self.alpha, bm25_hits), (self.alpha, dense_hits)):
            for rank, (device, chunk, _) in enumerate(hits):
                key = (device, chunk)
                chunk_map[key] = device
                rrf[key] = rrf.get(key, 0.0) + weight / (k + rank + 1)

        fused = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:pool]

        if self.reranker is not None:
            pairs = [(query, chunk) for (_, chunk), _ in fused]
            scores = self.reranker.predict(pairs)
            fused = sorted(
                [((device, chunk), float(s)) for ((device, chunk), _), s in zip(fused, scores)],
                key=lambda x: x[1], reverse=True,
            )

        top = [(device, chunk, score) for (device, chunk), score in fused[:top_k]]
        if top:
            lo = min(s for _, _, s in top)
            hi = max(s for _, _, s in top)
            span = (hi - lo) or 1.0
            top = [(d, c, (s - lo) / span) for d, c, s in top]
        return top


def _try_dense(documents: Dict[str, str], chunks: List[Chunk]) -> Optional[DenseRAG]:
    try:
        return DenseRAG(documents, chunk_cache=chunks)
    except Exception as e:  # ImportError / 模型下载失败 / 无网络
        print(f"[检索] 向量检索不可用，已降级：{e}")
        return None


def _try_reranker():
    if not config.RERANK:
        return None
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder(config.RERANK_MODEL)
    except Exception as e:
        print(f"[检索] 重排模型不可用，已跳过：{e}")
        return None


def build_retriever(documents: Dict[str, str], mode: Optional[str] = None):
    """按配置构造检索器；依赖缺失时自动降级到 bigram。"""
    mode = (mode or config.RETRIEVER or "bm25").lower()

    if mode == "bigram":
        return SimpleRAG(documents)

    if mode == "bm25":
        return BM25RAG(documents)

    if mode == "hybrid":
        bm25 = BM25RAG(documents)
        dense = _try_dense(documents, bm25.chunks)
        if dense is None:
            return bm25
        return HybridRAG(bm25, dense, alpha=config.HYBRID_ALPHA,
                         reranker=_try_reranker())

    print(f"[检索] 未知检索后端 {mode!r}，回退 bm25")
    return BM25RAG(documents)
