import asyncio
import json

from fastapi import Response
from fastapi.testclient import TestClient

from app.main import app
from app.routes.openai import rag
from app.schemas.openai import ChatCompletionRequest, ChatMessage
from app.services.rag_service import GenerationResult


class _StubRagService:
    async def generate_chat_completion(self, chat_req: ChatCompletionRequest):
        return GenerationResult(
            requested_model_id=chat_req.model,
            resolved_model_id="advanced-rag-agent-canonical",
            agent_output={
                "answer": "Hello from agent",
                "citations": [
                    {"asset_key": "doc-1", "page_number": 3, "source_url": "/tmp/doc.pdf"}
                ],
            },
        )


class _FailingService:
    async def generate_chat_completion(self, chat_req: ChatCompletionRequest):
        raise RuntimeError("boom")


class _UpstreamErrorService:
    async def generate_chat_completion(self, chat_req: ChatCompletionRequest):
        from app.core.exceptions import UpstreamModelError

        raise UpstreamModelError("Malformed upstream output")


def _make_request(**kwargs) -> ChatCompletionRequest:
    payload = {
        "model": "stale-openwebui-model",
        "messages": [ChatMessage(role="user", content="Hi")],
    }
    payload.update(kwargs)
    return ChatCompletionRequest(**payload)


def test_route_returns_markdown_when_requested():
    response = asyncio.run(
        rag(
            chat=_make_request(),
            response=Response(),
            x_response_format="markdown",
            x_request_id=None,
            service=_StubRagService(),
        )
    )

    content = response.choices[0].message.content
    assert "Hello from agent" in content
    assert "**Sources:**" in content
    assert "[doc-1 - Page 3](file:///tmp/doc.pdf)" in content
    assert response.model == "advanced-rag-agent-canonical"


def test_route_returns_json_string_when_requested():
    response = asyncio.run(
        rag(
            chat=_make_request(),
            response=Response(),
            x_response_format="json",
            x_request_id=None,
            service=_StubRagService(),
        )
    )

    content = response.choices[0].message.content
    parsed = json.loads(content)
    assert parsed["answer"] == "Hello from agent"
    assert parsed["citations"][0]["asset_key"] == "doc-1"
    assert response.model == "advanced-rag-agent-canonical"


def test_api_returns_openai_error_for_stream_true():
    client = TestClient(app)
    app.state.rag_service = _StubRagService()

    res = client.post(
        "/v1/chat/completions",
        headers={"x-request-id": "req-stream"},
        json={
            "model": "advanced-rag-agent",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        },
    )

    assert res.status_code == 200
    assert res.headers.get("x-request-id") == "req-stream"
    assert "text/event-stream" in res.headers["content-type"]

    data_lines = [line for line in res.text.splitlines() if line.startswith("data: ")]
    assert len(data_lines) == 4
    assert data_lines[-1] == "data: [DONE]"

    first = json.loads(data_lines[0].replace("data: ", ""))
    second = json.loads(data_lines[1].replace("data: ", ""))
    final = json.loads(data_lines[2].replace("data: ", ""))

    assert first["object"] == "chat.completion.chunk"
    assert first["choices"][0]["delta"]["role"] == "assistant"
    assert second["choices"][0]["delta"]["content"]
    assert final["choices"][0]["finish_reason"] == "stop"


def test_api_returns_openai_error_for_validation_failure():
    client = TestClient(app)
    app.state.rag_service = _StubRagService()

    res = client.post(
        "/v1/chat/completions",
        json={"model": "advanced-rag-agent", "messages": [{"role": "bad", "content": "Hello"}]},
    )

    assert res.status_code == 422
    body = res.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["code"] == "request_validation_error"


def test_api_returns_openai_error_for_internal_failures():
    client = TestClient(app)
    app.state.rag_service = _FailingService()

    res = client.post(
        "/v1/chat/completions",
        json={"model": "advanced-rag-agent", "messages": [{"role": "user", "content": "Hello"}]},
    )

    assert res.status_code == 500
    body = res.json()
    assert body["error"]["type"] == "server_error"
    assert body["error"]["code"] == "internal_server_error"


def test_api_returns_openai_error_for_upstream_model_failures():
    client = TestClient(app)
    app.state.rag_service = _UpstreamErrorService()

    res = client.post(
        "/v1/chat/completions",
        json={"model": "advanced-rag-agent", "messages": [{"role": "user", "content": "Hello"}]},
    )

    assert res.status_code == 502
    body = res.json()
    assert body["error"]["type"] == "upstream_model_error"
    assert body["error"]["code"] == "upstream_model_error"


def test_api_stream_honors_json_response_format():
    client = TestClient(app)
    app.state.rag_service = _StubRagService()

    res = client.post(
        "/v1/chat/completions",
        headers={"x-response-format": "json"},
        json={
            "model": "advanced-rag-agent",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        },
    )

    assert res.status_code == 200
    data_lines = [line for line in res.text.splitlines() if line.startswith("data: ")]
    second = json.loads(data_lines[1].replace("data: ", ""))
    streamed_content = second["choices"][0]["delta"]["content"]
    parsed = json.loads(streamed_content)
    assert parsed["answer"] == "Hello from agent"
