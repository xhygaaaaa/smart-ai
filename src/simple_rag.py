"""简化版RAG检索模块 - 基于bigram关键词匹配，不依赖向量数据库"""
import re
from typing import List, Tuple


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
        """将每份文档切分成段落"""
        for name, content in self.documents.items():
            # 按换行符切分段落，过滤空行
            chunks = [line.strip() for line in content.split('\n')
                     if line.strip() and len(line.strip()) > 10]
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

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
        """
        检索最相关的文档片段

        Args:
            query: 用户问题
            top_k: 返回前k个结果

        Returns:
            [(设备名称, 段落内容, 相似度得分), ...]
        """
        query_bigrams = self._extract_bigrams(query)

        results = []
        for device_name, chunks in self.doc_chunks.items():
            for chunk in chunks:
                chunk_bigrams = self._extract_bigrams(chunk)
                # 计算Jaccard相似度
                if len(query_bigrams) > 0:
                    intersection = len(query_bigrams & chunk_bigrams)
                    union = len(query_bigrams | chunk_bigrams)
                    score = intersection / union if union > 0 else 0

                    if score > 0:
                        results.append((device_name, chunk, score))

        # 按得分降序排序，取前top_k
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]
