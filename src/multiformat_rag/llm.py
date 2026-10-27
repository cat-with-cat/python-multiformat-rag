"""Configurable LLM adapters."""

from __future__ import annotations

import math
from typing import Protocol

import httpx

from llama_index.llms.dashscope import DashScope

from multiformat_rag.config import AppConfig
from multiformat_rag.exceptions import ConfigurationError, RAGError


class LLMClient(Protocol):
    """Minimal interface used by the RAG service and test doubles."""

    def complete(self, prompt: str) -> str:
        """Generate a complete answer for a prompt."""


class DashScopeLLMClient:
    """DashScope adapter that keeps provider details outside RAG logic."""

    def __init__(
        self,
        *,
        model_name: str,
        api_key: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> None:
        if not api_key:
            raise ValueError("DashScope API Key 不能为空。")
        self._llm = DashScope(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def complete(self, prompt: str) -> str:
        try:
            response = self._llm.complete(prompt)
        except Exception as exc:
            raise RAGError(
                "DashScope 调用失败。请检查网络、模型名称、账户额度和 API Key 配置。"
            ) from exc
        return response.text.strip()


class OpenAICompatibleLLMClient:
    """Minimal OpenAI Chat Completions adapter for compatible providers."""

    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        api_key: str,
        timeout_seconds: float = 30.0,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        http_client: httpx.Client | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("OpenAI 兼容接口 Base URL 不能为空。")
        if not model_name.strip():
            raise ValueError("OpenAI 兼容接口模型名称不能为空。")
        if not api_key:
            raise ValueError("OpenAI 兼容接口 API Key 不能为空。")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("LLM 请求超时必须大于 0 秒。")
        if max_tokens <= 0:
            raise ValueError("max_tokens 必须大于 0。")

        self._base_url = base_url.strip().rstrip("/")
        self._model_name = model_name.strip()
        self._api_key = api_key
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._http_client = http_client or httpx.Client(
            timeout=httpx.Timeout(timeout_seconds)
        )

    def complete(self, prompt: str) -> str:
        try:
            response = self._http_client.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self._temperature,
                    "max_tokens": self._max_tokens,
                },
            )
        except httpx.TimeoutException as exc:
            raise RAGError(
                "OpenAI 兼容 LLM 请求超时。请稍后重试或调整 LLM_TIMEOUT_SECONDS。"
            ) from exc
        except httpx.RequestError as exc:
            raise RAGError(
                "OpenAI 兼容 LLM 网络请求失败。请检查 Base URL 和网络连接。"
            ) from exc

        if response.status_code in {401, 403}:
            raise RAGError(
                "OpenAI 兼容 LLM 鉴权失败。请检查 API Key 和账户权限。"
            )
        if response.status_code == 404:
            raise RAGError(
                "OpenAI 兼容 LLM 端点或模型不可用。请核对 Base URL 和模型名称。"
            )
        if response.status_code == 429:
            raise RAGError(
                "OpenAI 兼容 LLM 请求受限。请检查账户额度、速率限制或稍后重试。"
            )
        if response.status_code >= 500:
            raise RAGError("OpenAI 兼容 LLM 服务暂时不可用，请稍后重试。")
        if response.status_code >= 400:
            raise RAGError(
                "OpenAI 兼容 LLM 请求被拒绝。请检查模型名称和请求配置。"
            )

        try:
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise RAGError("OpenAI 兼容 LLM 返回了无法解析的响应。") from exc
        if content is None:
            return ""
        if not isinstance(content, str):
            raise RAGError("OpenAI 兼容 LLM 返回了无法解析的响应。")
        return content.strip()


def create_llm_client(
    config: AppConfig,
    *,
    model_name: str | None = None,
) -> LLMClient:
    """Create the configured provider adapter while preserving safe key errors."""

    selected_model = model_name or config.llm_model
    if config.llm_provider == "dashscope":
        return DashScopeLLMClient(
            model_name=selected_model,
            api_key=config.require_dashscope_api_key(),
        )
    if config.llm_provider == "openai_compatible":
        return OpenAICompatibleLLMClient(
            base_url=config.llm_base_url,
            model_name=selected_model,
            api_key=config.require_openai_compatible_api_key(),
            timeout_seconds=config.llm_timeout_seconds,
        )
    raise ConfigurationError(
        "不支持的 LLM_PROVIDER。可选值：dashscope、openai_compatible。"
    )
