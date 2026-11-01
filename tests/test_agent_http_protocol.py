from __future__ import annotations

import json
from typing import Any

import httpx
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from multiformat_rag.agentic_rag import AGENT_MODEL_ERROR, create_rag_agent
from multiformat_rag.chat_model import create_agent_chat_model
from multiformat_rag.config import AppConfig
from multiformat_rag.langchain_integration import (
    SEARCH_TOOL_NAME,
    create_search_knowledge_base_tool,
)
from multiformat_rag.service import RAGService


def _config() -> AppConfig:
    return AppConfig(
        llm_provider="openai_compatible",
        llm_base_url="https://mock-gateway.invalid/v1",
        llm_model="mock-tool-model",
        llm_api_key="fake-protocol-test-key",
        llm_timeout_seconds=5.0,
    )


def _chat_response(message: dict[str, Any], finish_reason: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-offline-test",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "mock-tool-model",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }


class _EmptyIndex:
    def retrieve(self, question: str, *, top_k: int, min_similarity: float | None):
        return []


def test_chat_openai_sends_tool_schema_and_parses_tool_call() -> None:
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json=_chat_response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-http-1",
                            "type": "function",
                            "function": {
                                "name": SEARCH_TOOL_NAME,
                                "arguments": '{"query":"索引如何保存？"}',
                            },
                        }
                    ],
                },
                "tool_calls",
            ),
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        model = create_agent_chat_model(_config(), http_client=client)
        tool = create_search_knowledge_base_tool(RAGService(_EmptyIndex()))
        bound = model.bind_tools(
            [tool], tool_choice="auto", parallel_tool_calls=False
        )
        response = bound.invoke([HumanMessage(content="项目文档中索引如何保存？")])

    payload = captured[0]
    function = payload["tools"][0]["function"]
    assert payload["tool_choice"] == "auto"
    assert payload["parallel_tool_calls"] is False
    assert payload["tools"][0]["type"] == "function"
    assert function["name"] == SEARCH_TOOL_NAME
    assert function["parameters"]["required"] == ["query"]
    assert function["parameters"]["properties"]["query"]["type"] == "string"
    assert isinstance(response, AIMessage)
    assert response.tool_calls == [
        {
            "name": SEARCH_TOOL_NAME,
            "args": {"query": "索引如何保存？"},
            "id": "call-http-1",
            "type": "tool_call",
        }
    ]


def test_chat_openai_serializes_tool_message_with_matching_call_id() -> None:
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json=_chat_response(
                {"role": "assistant", "content": "Grounded answer [1]."},
                "stop",
            ),
            request=request,
        )

    prior_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": SEARCH_TOOL_NAME,
                "args": {"query": "index"},
                "id": "call-http-linked",
                "type": "tool_call",
            }
        ],
    )
    tool_message = ToolMessage(
        content='{"status":"ok","results":[]}',
        tool_call_id="call-http-linked",
        name=SEARCH_TOOL_NAME,
    )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        model = create_agent_chat_model(_config(), http_client=client)
        response = model.invoke(
            [
                HumanMessage(content="Where is the index?"),
                prior_call,
                tool_message,
            ]
        )

    messages = captured[0]["messages"]
    assert messages[1]["role"] == "assistant"
    assert messages[1]["tool_calls"][0]["id"] == "call-http-linked"
    assert messages[2] == {
        "role": "tool",
        "content": '{"status":"ok","results":[]}',
        "tool_call_id": "call-http-linked",
    }
    assert isinstance(response, AIMessage)
    assert response.content == "Grounded answer [1]."


def test_http_error_is_sanitized_by_agent_without_retry() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            500,
            json={
                "error": {
                    "message": "internal failure at /private/server/path",
                    "type": "server_error",
                }
            },
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        model = create_agent_chat_model(_config(), http_client=client)
        agent = create_rag_agent(RAGService(_EmptyIndex()), chat_model=model)
        output = agent.invoke({"question": "general question"})

    assert len(requests) == 1
    assert output["status"] == "error"
    assert output["answer"] == AGENT_MODEL_ERROR
    assert "/private/server/path" not in str(output)
    assert "fake-protocol-test-key" not in str(output)
