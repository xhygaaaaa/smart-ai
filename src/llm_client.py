"""LLM大模型调用模块 - 支持OpenAI兼容API"""
import time
from typing import List, Tuple, Optional
from openai import OpenAI
import config


class LLMClient:
    """大模型调用客户端"""

    def __init__(self):
        """初始化OpenAI客户端"""
        self.client = OpenAI(
            base_url=config.LLM_BASE_URL,
            api_key=config.LLM_API_KEY
        )
        self.model = config.LLM_MODEL
        self.temperature = config.LLM_TEMPERATURE

    def _call_api_with_retry(self, messages: list, max_retries: int = 3) -> str:
        """
        调用API并自动重试(处理429限流)

        Args:
            messages: 对话消息列表
            max_retries: 最大重试次数

        Returns:
            AI回答文本
        """
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature
                )
                return response.choices[0].message.content

            except Exception as e:
                error_str = str(e)
                # 检测429限流错误
                if "429" in error_str or "rate" in error_str.lower():
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 3  # 3秒、6秒、9秒指数退避
                        print(f"遇到限流，等待{wait_time}秒后重试...")
                        time.sleep(wait_time)
                        continue
                raise e

        raise Exception("API调用失败，已达最大重试次数")

    def answer_question(
        self,
        question: str,
        retrieved_docs: List[Tuple[str, str, float]]
    ) -> str:
        """
        基于检索文档生成答案

        Args:
            question: 用户问题
            retrieved_docs: 检索到的文档片段 [(设备名, 内容, 得分), ...]

        Returns:
            AI生成的回答
        """
        # 构建上下文
        context_parts = []
        for i, (device, content, score) in enumerate(retrieved_docs, 1):
            context_parts.append(f"[文档{i} - {device}]\n{content}")

        context = "\n\n".join(context_parts)

        # 构建prompt
        system_prompt = """你是SmartGuide AI客服助手，专门回答智能硬件使用问题。

回答规则：
1. 只根据提供的文档内容回答，不能编造信息
2. 如果文档中没有相关信息，明确告知"该信息不在设备使用指南中，无法作答"
3. 回答要简洁准确，突出关键步骤和数字
4. 对于超出使用指南的问题(如价格、保修、购买渠道、维修、对比建议)，礼貌拒答
"""

        user_prompt = f"""参考文档：
{context}

用户问题：{question}

请根据参考文档回答用户问题。"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        return self._call_api_with_retry(messages)
