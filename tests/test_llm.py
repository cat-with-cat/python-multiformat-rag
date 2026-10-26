import json
from pathlib import Path

import httpx
import pytest

from multiformat_rag.cli import main
from multiformat_rag.config import AppConfig
from multiformat_rag.exceptions import ConfigurationError, RAGError
from multiformat_rag.llm import (
    DashScopeLLMClient,
    OpenAICompatibleLLMClient,
    create_llm_client,
)


def _http_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_openai_compatible_client_uses_custom_endpoint_model_and_bearer() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["Authorization"]
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "  有依据的回答 [1]  "}}]},
        )

    client = OpenAICompatibleLLMClient(
        base_url="https://gateway.example/v1/",
        model_name="available-model-id",
        api_key="test-only-secret",
        http_client=_http_client(handler),
    )

    assert client.complete("grounded prompt") == "有依据的回答 [1]"
    assert captured["url"] == "https://gateway.example/v1/chat/completions"
    assert captured["authorization"] == "Bearer test-only-secret"
    assert captured["payload"] == {
        "model": "available-model-id",
        "messages": [{"role": "user", "content": "grounded prompt"}],
        "temperature": 0.1,
        "max_tokens": 1024,
    }


@pytest.mark.parametrize(
    ("status_code", "message"),
    [
        (401, "鉴权失败"),
        (403, "鉴权失败"),
        (404, "端点或模型不可用"),
        (429, "请求受限"),
        (500, "服务暂时不可用"),
        (400, "请求被拒绝"),
    ],
)
def test_openai_compatible_http_errors_are_safe(
    status_code: int,
    message: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            json={"error": {"message": "secret-response-detail"}},
        )

    client = OpenAICompatibleLLMClient(
        base_url="https://gateway.example/v1",
        model_name="test-model",
        api_key="secret-key-value",
        http_client=_http_client(handler),
    )

    with pytest.raises(RAGError, match=message) as captured:
        client.complete("prompt")

    assert "secret-key-value" not in str(captured.value)
    assert "secret-response-detail" not in str(captured.value)


def test_openai_compatible_timeout_is_wrapped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("provider detail", request=request)

    client = OpenAICompatibleLLMClient(
        base_url="https://gateway.example/v1",
        model_name="test-model",
        api_key="secret-key-value",
        http_client=_http_client(handler),
    )

    with pytest.raises(RAGError, match="请求超时") as captured:
        client.complete("prompt")

    assert "provider detail" not in str(captured.value)


def test_openai_compatible_empty_answer_is_returned_for_service_fallback() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "   "}}]},
        )

    client = OpenAICompatibleLLMClient(
        base_url="https://gateway.example/v1",
        model_name="test-model",
        api_key="test-key",
        http_client=_http_client(handler),
    )

    assert client.complete("prompt") == ""


def test_openai_compatible_malformed_response_is_wrapped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    client = OpenAICompatibleLLMClient(
        base_url="https://gateway.example/v1",
        model_name="test-model",
        api_key="test-key",
        http_client=_http_client(handler),
    )

    with pytest.raises(RAGError, match="无法解析"):
        client.complete("prompt")


@pytest.mark.parametrize("timeout_seconds", [0, -1, float("inf"), float("nan")])
def test_openai_compatible_rejects_invalid_timeout(
    timeout_seconds: float,
) -> None:
    with pytest.raises(ValueError, match="超时"):
        OpenAICompatibleLLMClient(
            base_url="https://gateway.example/v1",
            model_name="test-model",
            api_key="test-key",
            timeout_seconds=timeout_seconds,
        )


def test_openai_compatible_factory_uses_environment_configuration(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.openlux.ai/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "account-model-id")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "12.5")
    config = AppConfig.from_env(tmp_path / "missing.env")

    client = create_llm_client(config)

    assert isinstance(client, OpenAICompatibleLLMClient)
    assert client._base_url == "https://api.openlux.ai/v1"
    assert client._model_name == "account-model-id"


def test_factory_preserves_dashscope_provider() -> None:
    config = AppConfig(
        llm_provider="dashscope",
        llm_model="qwen-turbo",
        dashscope_api_key="test-only-key",
    )

    client = create_llm_client(config)

    assert isinstance(client, DashScopeLLMClient)


def test_openai_compatible_missing_key_is_safe(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    config = AppConfig.from_env(tmp_path / "missing.env")

    with pytest.raises(ConfigurationError, match="LLM_API_KEY"):
        create_llm_client(config)


def test_unknown_llm_provider_is_rejected(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "unknown-provider")
    config = AppConfig.from_env(tmp_path / "missing.env")

    with pytest.raises(ConfigurationError, match="LLM_PROVIDER"):
        create_llm_client(config)


def test_openai_compatible_cli_checks_key_before_loading_index(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    exit_code = main(
        [
            "--env-file",
            str(tmp_path / "missing.env"),
            "ask",
            "question",
            "--persist-dir",
            str(tmp_path / "missing-index"),
        ]
    )

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "LLM_API_KEY" in captured.err
    assert "missing-index" not in captured.err
