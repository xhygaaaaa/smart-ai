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
        # 用户口语“充不进电/充不上电”与手册里的“无法开机/连接充电器”字面不重合，
        # 不做同义扩展会把正确的故障排查段落漏掉（实测 top-10 都进不去）。
        "充不进电": "无法充电 充电故障 充电触点 充电器",
        "充不进": "无法充电 充电故障 充电触点",
        "充不上电": "无法充电 充电故障 充电触点 充电器",
        "无法充电": "充电故障 充电触点 充电器",
        "不充电": "无法充电 充电故障 充电触点",
        "怎么充电": "充电方法 连接充电器 充电触点 充电时间",
        "如何充电": "充电方法 连接充电器 充电触点 充电时间",
        "充电": "充电器 充电触点 充电时间",
        # 「重置」是用户口语，手册里写的是「恢复出厂设置」，字面完全不重合
        "怎么重置": "恢复出厂设置 重新配网 重新添加设备",
        "重置": "恢复出厂设置 重新配网",
        "复位": "恢复出厂设置 重启",
        "恢复出厂": "恢复出厂设置 重新配网",
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
        # 设备定向：问题里出现设备关键词时，只检索该设备的手册，避免跨设备噪音
        target_device = None
        device_word = None
        for kw, name in DEVICE_KEYWORD_MAP.items():
            if kw in query:
                target_device = name
                device_word = kw
                break

        # 已经按设备过滤过了，“智能/手环”这些词对区分段落没有信息量，
        # 留在查询里只会让“产品概述”这种泛泛的段落拿高分。
        term_query = query.replace(device_word, "") if device_word else query
        query_bigrams = self._extract_bigrams(self._expand_query(term_query))

        results = []
        for device_name, chunks in self.doc_chunks.items():
            if target_device and device_name != target_device:
                continue
            for chunk in chunks:
                chunk_bigrams = self._extract_bigrams(chunk)
                if not query_bigrams:
                    continue
                intersection = len(query_bigrams & chunk_bigrams)
                if intersection == 0:
                    continue

                # 原来的分母用 min(查询, 段落)，等于奖励“短段落”：
                # “产品概述”这种只有两三行的块几乎稳拿第一，把真正的
                # “技术规格/电池续航”挤下去。改成以“查询覆盖率”为主：
                #   覆盖率 = 命中的查询词 / 查询总词数
                #   精确率 = 命中的查询词 / 段落词数（权重低，避免长段落吃亏）
                coverage = intersection / len(query_bigrams)
                precision = intersection / max(1, len(chunk_bigrams))

                # 小节标题命中额外加分（“电池续航是多少天”应优先落在“续航时间”段）
                heading = chunk.split("\n", 1)[0]
                heading_bigrams = self._extract_bigrams(heading)
                heading_hit = (
                    len(query_bigrams & heading_bigrams) / len(query_bigrams)
                    if heading_bigrams else 0
                )

                score = 0.55 * coverage + 0.20 * precision + 0.25 * heading_hit
                if score > 0:
                    results.append((device_name, chunk, score))

        # 按得分降序排序，取前top_k
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]
