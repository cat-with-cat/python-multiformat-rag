"""Environment-backed application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from multiformat_rag.exceptions import ConfigurationError

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_LLM_MODEL = "qwen-turbo"
DEFAULT_LLM_PROVIDER = "dashscope"
DEFAULT_OPENAI_COMPATIBLE_BASE_URL = "https://api.openlux.ai/v1"
DEFAULT_LLM_TIMEOUT_SECONDS = 30.0


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ConfigurationError(f"环境变量 {name} 必须是整数。") from exc


def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ConfigurationError(f"环境变量 {name} 必须是数字。") from exc


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Runtime settings shared by the CLI and application services."""

    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    llm_provider: str = DEFAULT_LLM_PROVIDER
    llm_model: str = DEFAULT_LLM_MODEL
    llm_base_url: str = DEFAULT_OPENAI_COMPATIBLE_BASE_URL
    llm_api_key: str | None = None
    llm_timeout_seconds: float = DEFAULT_LLM_TIMEOUT_SECONDS
    index_dir: Path = Path(".rag_index")
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    min_similarity: float = 0.35
    dashscope_api_key: str | None = None

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> "AppConfig":
        """Load optional ``.env`` values without ever logging secret contents."""

        load_dotenv(dotenv_path=env_file, override=False)
        llm_model = (
            os.getenv("LLM_MODEL")
            or os.getenv("RAG_LLM_MODEL")
            or DEFAULT_LLM_MODEL
        )
        return cls(
            embedding_model=os.getenv(
                "RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
            ),
            llm_provider=(
                os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER).strip().lower()
                or DEFAULT_LLM_PROVIDER
            ),
            llm_model=llm_model,
            llm_base_url=(
                os.getenv(
                    "LLM_BASE_URL", DEFAULT_OPENAI_COMPATIBLE_BASE_URL
                ).strip()
                or DEFAULT_OPENAI_COMPATIBLE_BASE_URL
            ),
            llm_api_key=os.getenv("LLM_API_KEY") or None,
            llm_timeout_seconds=_env_float(
                "LLM_TIMEOUT_SECONDS", DEFAULT_LLM_TIMEOUT_SECONDS
            ),
            index_dir=Path(os.getenv("RAG_INDEX_DIR", ".rag_index")),
            chunk_size=_env_int("RAG_CHUNK_SIZE", 512),
            chunk_overlap=_env_int("RAG_CHUNK_OVERLAP", 64),
            top_k=_env_int("RAG_TOP_K", 5),
            min_similarity=_env_float("RAG_MIN_SIMILARITY", 0.35),
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY") or None,
        )

    def require_dashscope_api_key(self) -> str:
        """Return the API key or raise a safe error that does not reveal it."""

        if not self.dashscope_api_key:
            raise ConfigurationError(
                "未配置 DASHSCOPE_API_KEY。请复制 .env.example 为 .env，"
                "填入合法密钥后再使用 ask 命令。retrieve 命令无需 API Key。"
            )
        return self.dashscope_api_key

    def require_openai_compatible_api_key(self) -> str:
        """Return the compatible API key without exposing its value."""

        if not self.llm_api_key:
            raise ConfigurationError(
                "未配置 LLM_API_KEY。使用 openai_compatible provider 时，请在"
                "本地 .env 中填入合法密钥；不要将 .env 提交到 Git。"
            )
        return self.llm_api_key
