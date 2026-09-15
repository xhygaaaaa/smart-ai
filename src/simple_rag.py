"""简化版RAG检索模块 - 基于bigram关键词匹配，不依赖向量数据库"""
import re
from typing import List, Tuple


# 设备关键词 -> 手册名称 映射，用于把问题定向到对应设备的知识库
DEVICE_KEYWORD_MAP = {
    "手环": "智能手环",
    "手表": "智能手表",
    "音箱": "智能音箱",
    "台灯": "智能台灯",
    "门锁": "智能门锁",
}


class SimpleRAG:
    """基于关键词匹配的简单RAG检索"""

    def __init__(self, documents: dict):
        """
        初始化RAG检索器

        Args:
            documents: {设备名称: 文档内容} 的字典
        """
        self.documents = documents
        self.doc_chunks = {}  # {设备名称: [段落列表]}
        self._build_index()

    def _build_index(self):
        """将每份文档切分成段落(按空行切分，保留完整章节，避免标题与步骤被拆散)"""
        for name, content in self.documents.items():
            # 按空行切分成完整章节段落(标题 + 若干步骤作为一个整体)，过滤过短片段
            chunks = [block.strip() for block in re.split(r'\n\s*\n', content)
                     if block.strip() and len(block.strip()) > 10]
            self.doc_chunks[name] = chunks

    def _extract_bigrams(self, text: str) -> set:
        """
        提取文本的bigram(2字组合)作为关键词

        Args:
            text: 输入文本

        Returns:
            bigram集合
        """
        # 只保留中文、数字、字母
        clean_text = re.sub(r'[^一-龥a-zA-Z0-9]', '', text)
        bigrams = set()
        for i in range(len(clean_text) - 1):
            bigram = clean_text[i:i+2]
            if len(bigram) == 2:
                bigrams.add(bigram)
        return bigrams

    # 口语化问题 -> 正式术语 的同义扩展，弥补 bigram 无法理解"连=连接=配对"的语义鸿沟
    QUERY_SYNONYMS = {
        "连手机": "连接手机 配对 蓝牙",
        "连不上": "连接 配对 蓝牙",
        "怎么连": "怎么连接 怎么配对 蓝牙",
    }

    def _expand_query(self, query: str) -> str:
        """把口语词扩展成正式术语后再提取 bigram，提升召回"""
        expanded = query
        for kw, extra in self.QUERY_SYNONYMS.items():
            if kw in query:
                expanded += " " + extra
        return expanded

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
        """
        检索最相关的文档片段

        Args:
            query: 用户问题
            top_k: 返回前k个结果

        Returns:
            [(设备名称, 段落内容, 相似度得分), ...]
        """
        query_bigrams = self._extract_bigrams(self._expand_query(query))

        # 设备定向：问题里出现设备关键词时，只检索该设备的手册，避免跨设备噪音
        target_device = None
        for kw, name in DEVICE_KEYWORD_MAP.items():
            if kw in query:
                target_device = name
                break

        results = []
        for device_name, chunks in self.doc_chunks.items():
            if target_device and device_name != target_device:
                continue
            for chunk in chunks:
                chunk_bigrams = self._extract_bigrams(chunk)
                # 用 overlap 系数(交集/较短集合)替代 Jaccard，避免长段落被过度惩罚
                if len(query_bigrams) > 0:
                    intersection = len(query_bigrams & chunk_bigrams)
                    denom = min(len(query_bigrams), len(chunk_bigrams))
                    score = intersection / denom if denom > 0 else 0

                    if score > 0:
                        results.append((device_name, chunk, score))

        # 按得分降序排序，取前top_k
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]
