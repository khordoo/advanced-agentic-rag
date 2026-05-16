import asyncio

from langchain_core.messages import AIMessage

from app.schemas.openai import ChatCompletionRequest, ChatMessage
from app.services.rag_service import RagService


class _StubGraph:
    async def ainvoke(self, payload):
        return {
            "messages": [
                AIMessage(
                    content='{"answer":"Canonical answer","citations":[{"asset_key":"doc-7"}]}'
                )
            ]
        }


def test_rag_service_returns_canonical_agent_payload():
    service = RagService(graph=_StubGraph(), system_instructions="system", model_id="resolved-id")
    request = ChatCompletionRequest(
        model="stale-client-id",
        messages=[ChatMessage(role="user", content="question?")],
    )

    output = asyncio.run(service.generate_chat_completion(request))

    assert output.requested_model_id == "stale-client-id"
    assert output.resolved_model_id == "resolved-id"
    assert output.agent_output["answer"] == "Canonical answer"
    assert output.agent_output["citations"][0]["asset_key"] == "doc-7"
