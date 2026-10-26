"""评测模块 - 实现三级分类和方案A准确率算法"""
import re
from typing import Tuple, List
import config


class Evaluator:
    """RAG系统评测器"""

    def __init__(self):
        """初始化评测器"""
        self.refusal_signals = config.REFUSAL_SIGNALS

    def is_valid_question(self, question: str) -> bool:
        """
        判断问题是否有效(第一级:过滤完全不沾边的无效问题)

        Args:
            question: 用户问题

        Returns:
            是否有效
        """
        # 检查长度
        if len(question.strip()) < 3:
            return False

        # 明显偏离客服场景的话题：即使句子里带“怎么/什么”，也不该放行。
        # （例如“今天天气怎么样”会因为“怎么”被动作词规则误判为有效问题。）
        off_topic = ["天气", "股票", "新闻", "写诗", "写代码", "编程",
                     "翻译", "讲笑话", "电影推荐", "旅游攻略", "星座"]
        if any(kw in question for kw in off_topic):
            return False

        # 检查是否包含设备关键词
        device_keywords = ["手环", "手表", "音箱", "台灯", "门锁", "智能"]
        has_device = any(kw in question for kw in device_keywords)

        # 检查是否包含动作词
        action_keywords = ["如何", "怎么", "怎样", "什么", "多少", "哪里",
                          "能否", "可以", "支持", "是否", "为什么"]
        has_action = any(kw in question for kw in action_keywords)

        return has_device or has_action

    # 拒答判定阈值：短答案直接看话术；长答案看“拒答是不是开场就说的”。
    REFUSAL_MAX_LEN = 30
    REFUSAL_MIN_SUBSTANCE = 40

    def is_refusal(self, answer: str) -> bool:
        """
        判断答案是否为拒答(第二级:识别超纲问题)

        不能见到拒答词就判超纲：`不包含/无法确认/建议咨询` 这类词在正常回答里
        也常见，而模型经常**先答正文、末尾再补一句**“如需具体步骤，此处无法作答”。
        实测一个正经的功能类问答（智能音箱场景模式）因此被误判成超纲，
        连带把它从 groundedness / correct_rate 的统计里剔除。

        判定口径：
        - 短答案（<= REFUSAL_MAX_LEN 字）出现拒答话术即算拒答；
        - 长答案看**开场第一句**：一上来就说“不在指南范围内/无法作答”的算拒答
          （后面再补选购建议也还是拒答，如“哪个好”这类对比题）；
        - 拒答话术只出现在正文之后、且正文本身有实质内容（>= 40 字）的，不算拒答。

        Args:
            answer: AI回答

        Returns:
            是否拒答
        """
        text = (answer or "").strip()
        if not text:
            return False
        if not any(signal in text for signal in self.refusal_signals):
            return False
        if len(text) <= self.REFUSAL_MAX_LEN:
            return True

        sentences = [s for s in re.split(r"(?<=[。；！？!?;\n])", text) if s.strip()]
        first = sentences[0] if sentences else text
        if any(signal in first for signal in self.refusal_signals):
            return True

        kept = "".join(
            sentence for sentence in sentences
            if not any(signal in sentence for signal in self.refusal_signals)
        ).strip()
        return len(kept) < self.REFUSAL_MIN_SUBSTANCE

    def extract_keywords_v2(self, text: str) -> set:
        """
        改进版关键词提取(避免粗暴2字滑窗产生垃圾词)

        提取逻辑:
        1. 先提取"数字+单位"组合(如"7天""5ATM")
        2. 按标点和连接词切分短语
        3. 保留2-5字的有意义中文短语

        Args:
            text: 输入文本

        Returns:
            关键词集合
        """
        keywords = set()

        # 1. 提取"数字+单位"组合
        number_patterns = [
            r'\d+[天日小时分钟秒]',  # 时间: 7天、2小时
            r'\d+ATM',  # 防水等级: 5ATM
            r'\d+米',  # 距离: 5米
            r'\d+[KkMm]',  # 色温/存储: 2700K、100M
            r'\d+%',  # 百分比: 50%
            r'\d+-\d+[KkMm位]',  # 范围: 2700-6500K、6-12位
            r'\d+枚',  # 指纹数: 100枚
            r'\d+\.?\d*秒',  # 小数秒: 0.3秒
        ]
        for pattern in number_patterns:
            matches = re.findall(pattern, text)
            keywords.update(matches)

        # 2. 按标点和连接词切分短语
        # 移除常见连接词和语气词
        clean_text = re.sub(r"""[，。！？、；：""''（）()《》\[\]]""", ' ', text)
        filler_words = ['的', '了', '是', '在', '和', '与', '或', '即可',
                       '可以', '需要', '然后', '接着', '约', '大约']
        for word in filler_words:
            clean_text = clean_text.replace(word, ' ')

        # 3. 提取2-5字中文短语
        phrases = re.findall(r'[一-龥]{2,5}', clean_text)
        # 过滤掉常见停用词
        stopwords = {'然后', '接着', '如果', '这个', '那个', '进行', '开始'}
        meaningful_phrases = [p for p in phrases if p not in stopwords]
        keywords.update(meaningful_phrases)

        return keywords

    def calculate_accuracy_a(
        self,
        answer: str,
        retrieved_docs: List[Tuple[str, str, float]]
    ) -> Tuple[float, dict]:
        """
        方案A准确率:衡量"AI回答的关键词有多少能在引用文档中找到依据"

        这不是跟标准答案比对，而是衡量防幻觉能力

        Args:
            answer: AI生成的回答
            retrieved_docs: 检索到的文档 [(设备名, 内容, 得分), ...]

        Returns:
            (准确率, 详细信息字典)
        """
        # 提取答案关键词
        answer_keywords = self.extract_keywords_v2(answer)

        if len(answer_keywords) == 0:
            return 0.0, {
                "total_keywords": 0,
                "supported_keywords": 0,
                "unsupported_keywords": [],
                "answer_keywords": []
            }

        # 合并所有检索文档的内容
        all_doc_content = " ".join([content for _, content, _ in retrieved_docs])
        doc_keywords = self.extract_keywords_v2(all_doc_content)

        # 计算有多少答案关键词能在文档中找到
        supported = answer_keywords & doc_keywords
        unsupported = answer_keywords - doc_keywords

        accuracy = len(supported) / len(answer_keywords)

        return accuracy, {
            "total_keywords": len(answer_keywords),
            "supported_keywords": len(supported),
            "unsupported_keywords": list(unsupported),
            "answer_keywords": list(answer_keywords)
        }

    def calculate_correctness(
        self,
        answer: str,
        expected_answer: str,
        hit_threshold: float = 0.6,
    ) -> Tuple[float, dict]:
        """与标准答案比对的关键点召回率（correctness）。

        与 calculate_accuracy_a 的区别：
          - accuracy_a / groundedness：答案有没有被检索文档支撑（防幻觉）；
          - correctness：答案有没有覆盖标准答案的关键信息（答得对不对）。

        这里用关键词集合做轻量代理，不引入额外模型；需要更严格时
        可在评测脚本里用 LLM judge（见 src/evaluation_runner.py）。

        Returns:
            (关键点召回率, 详细信息)
        """
        # 用字符 bigram 而不是 2-5 字滑窗短语：后者会因切分边界不同
        # 把“长按侧边电源键3秒开机”切成“长按侧边电/源键”，召回偏低。
        ans_kw = self._char_bigrams(answer or "")
        exp_kw = self._char_bigrams(expected_answer or "")

        if not exp_kw:
            return 0.0, {"error": "标准答案为空或无有效关键词", "hit": False}

        covered = ans_kw & exp_kw
        recall = len(covered) / len(exp_kw)
        precision = len(covered) / len(ans_kw) if ans_kw else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        return recall, {
            "hit": recall >= hit_threshold,
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "expected_keywords": len(exp_kw),
            "covered_keywords": len(covered),
            "missing_keywords": sorted(exp_kw - ans_kw)[:20],
        }

    @staticmethod
    def _char_bigrams(text: str) -> set:
        """去标点后的字符 bigram 集合（中文按字，英文数字保留整词）。"""
        clean = re.sub(r'[^一-龥a-zA-Z0-9]', '', text or "").lower()
        grams = {clean[i:i + 2] for i in range(len(clean) - 1)}
        for word in re.findall(r'[a-z0-9]{2,}', clean):
            grams.add(word)
        return grams

    def classify_question(
        self,
        question: str,
        answer: str
    ) -> Tuple[str, str]:
        """
        三级问题分类

        Args:
            question: 用户问题
            answer: AI回答

        Returns:
            (分类, 说明文字)
        """
        # 第一级:无效问题
        if not self.is_valid_question(question):
            return "无效", "问题不符合客服场景(过短或不包含设备关键词)"

        # 第二级:超纲问题
        if self.is_refusal(answer):
            return "超纲", "AI拒答(问题超出知识库范围，如价格/保修/购买)"

        # 第三级:有效问题
        return "有效", "正常回答，计入准确率统计"
