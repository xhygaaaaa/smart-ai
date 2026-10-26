"""LLM大模型调用模块 - 支持OpenAI兼容API"""
import time
from typing import Dict, List, Tuple, Optional
from openai import OpenAI
import config

SYSTEM_PROMPT = """你是SmartGuide AI客服助手，专门回答智能硬件使用问题。

回答规则：
1. 只根据提供的文档内容回答，不能编造信息
2. 如果文档中没有相关信息，明确告知"该信息不在设备使用指南中，无法作答"
3. 回答要简洁准确，突出关键步骤和数字
4. 对于超出使用指南的问题(如价格、保修、购买渠道、维修、对比建议)，礼貌拒答
5. 如果「对话历史」里用户已经说明过设备或问题背景，可以结合历史理解当前问题，但事实仍必须以参考文档为准
"""


def build_messages(
    question: str,
    retrieved_docs: List[Tuple[str, str, float]],
    history: Optional[List[Dict[str, str]]] = None,
    max_turns: Optional[int] = None,
) -> List[Dict[str, str]]:
    """构造发给模型的 messages。

    多轮策略：只把最近 max_turns 轮「用户问题 / 助手回答」放进上下文，
    参考文档只放当前这一轮的，避免历史文档把窗口挤爆。
    """
    max_turns = config.HISTORY_TURNS if max_turns is None else max_turns

    context_parts = []
    for i, (device, content, score) in enumerate(retrieved_docs, 1):
        context_parts.append(f"[文档{i} - {device}]\n{content}")
    context = "\n\n".join(context_parts)

    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    if history and max_turns > 0:
        # history 是 [{"role","content"}]，按轮截取最近 N 轮
        recent = history[-max_turns * 2:]
        for msg in recent:
            role = msg.get("role")
            content = msg.get("content")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    user_prompt = f"""参考文档：
{context}

用户问题：{question}

请根据参考文档回答用户问题。"""
    messages.append({"role": "user", "content": user_prompt})
    return messages


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

    def complete(self, messages: list, temperature: Optional[float] = None) -> str:
        """通用补全入口（评测 judge / 其他模块复用）。"""
        return self._call_api_with_retry(
            messages,
            temperature=self.temperature if temperature is None else temperature,
        )

    def _call_api_with_retry(self, messages: list, max_retries: int = 3,
                             temperature: Optional[float] = None) -> str:
        """
        调用API并自动重试(处理429限流)

        Args:
            messages: 对话消息列表
            max_retries: 最大重试次数
            temperature: 覆盖默认温度

        Returns:
            AI回答文本
        """
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature if temperature is None else temperature,
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
        retrieved_docs: List[Tuple[str, str, float]],
        history: Optional[List[Dict[str, str]]] = None,
        max_turns: Optional[int] = None,
    ) -> str:
        """
        基于检索文档生成答案（可选带多轮历史）

        Args:
            question: 用户问题
            retrieved_docs: 检索到的文档片段 [(设备名, 内容, 得分), ...]
            history: 历史对话 [{"role": "user"/"assistant", "content": ...}]
            max_turns: 最多带入几轮历史，默认取 config.HISTORY_TURNS

        Returns:
            AI生成的回答
        """
        messages = build_messages(question, retrieved_docs, history, max_turns)
        return self._call_api_with_retry(messages)
