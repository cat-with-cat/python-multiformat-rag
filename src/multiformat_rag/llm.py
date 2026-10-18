"""Configurable LLM adapters."""

from __future__ import annotations

from typing import Protocol

from llama_index.llms.dashscope import DashScope

from multiformat_rag.exceptions import RAGError


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
