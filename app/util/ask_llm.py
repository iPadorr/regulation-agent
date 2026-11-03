import asyncio

from openai import AsyncOpenAI
from app.config.conf import llm_config


async def ask_llm(name: str, messages, timeout: float = 10000.0):
    """
    异步请求 LLM
    1）未超时：返回补全内容
    2）超时：抛出超时错误
    """

    # 从配置中获取指定模型的配置信息
    model_config = llm_config.models[name]

    # 创建异步 OpenAI 客户端
    client = AsyncOpenAI(
        base_url=model_config.base_url,  # API 基础URL
        api_key=model_config.api_key,  # API 密钥
    )

    async def create_completion():
        """
        发起聊天补全请求（client.chat.completions.create），获取模型返回的响应内容
        """

        # 调用 OpenAI Chat Completions API
        completion = await client.chat.completions.create(
            model=model_config.model,  # 指定使用的模型
            messages=messages,  # 对话消息列表
            **model_config.params,  # 其他模型参数(如温度、最大token等)
        )
        return completion.choices[0].message.content

    try:
        # 使用 asyncio.wait_for 设置超时限制
        return await asyncio.wait_for(create_completion(), timeout=timeout)
    except asyncio.TimeoutError:
        # 超时则抛出错误信息
        raise TimeoutError(f"LLM request timed out after {timeout}s")