import asyncio
import json

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


def _make_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="stale-openwebui-model",
        messages=[ChatMessage(role="user", content="Hi")],
    )


def test_route_returns_markdown_when_requested():
    response = asyncio.run(
        rag(chat=_make_request(), x_response_format="markdown", service=_StubRagService())
    )

    content = response.choices[0].message.content
    assert "Hello from agent" in content
    assert "**Sources:**" in content
    assert "[doc-1 - Page 3](file:///tmp/doc.pdf)" in content
    assert response.model == "advanced-rag-agent-canonical"


def test_route_returns_json_string_when_requested():
    response = asyncio.run(
        rag(chat=_make_request(), x_response_format="json", service=_StubRagService())
    )

    content = response.choices[0].message.content
    parsed = json.loads(content)
    assert parsed["answer"] == "Hello from agent"
    assert parsed["citations"][0]["asset_key"] == "doc-1"
    assert response.model == "advanced-rag-agent-canonical"
