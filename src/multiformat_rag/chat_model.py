"""Chat-model factory used only by the optional V0.4 agent path."""

from __future__ import annotations

import math

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from multiformat_rag.config import AppConfig
from multiformat_rag.exceptions import ConfigurationError


def create_agent_chat_model(
    config: AppConfig,
    *,
    temperature: float = 0.0,
    max_completion_tokens: int = 1_024,
    http_client: httpx.Client | None = None,
) -> BaseChatModel:
    """Create a tool-capable chat model without changing the V0.3 LLM client.

    The returned model is configured for the Chat Completions protocol.  Client
    construction is local; no request is sent until the model is invoked.
    """

    if config.llm_provider != "openai_compatible":
        raise ConfigurationError(
            "当前 Agent 仅支持具备标准 Tool Calling 的 openai_compatible "
            "Provider；DashScope 文本生成路径仍可继续使用。"
        )
    if not config.llm_base_url.strip():
        raise ConfigurationError("Agent 的 LLM_BASE_URL 不能为空。")
    if not config.llm_model.strip():
        raise ConfigurationError("Agent 的 LLM_MODEL 不能为空。")
    if (
        not math.isfinite(config.llm_timeout_seconds)
        or config.llm_timeout_seconds <= 0
    ):
        raise ConfigurationError("Agent 的 LLM_TIMEOUT_SECONDS 必须大于 0。")
    if max_completion_tokens <= 0:
        raise ValueError("max_completion_tokens 必须大于 0。")

    return ChatOpenAI(
        model=config.llm_model.strip(),
        api_key=config.require_openai_compatible_api_key(),
        base_url=config.llm_base_url.strip().rstrip("/"),
        timeout=config.llm_timeout_seconds,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        max_retries=0,
        streaming=False,
        use_responses_api=False,
        http_client=http_client,
    )
