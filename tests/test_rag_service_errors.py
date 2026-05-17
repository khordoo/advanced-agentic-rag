import pytest

from app.core.exceptions import UpstreamModelError
from app.schemas.openai import ChatCompletionRequest, ChatMessage
from app.services.rag_service import RagService


class _Message:
    def __init__(self, content):
        self.content = content


class _GraphMissingMessages:
    async def ainvoke(self, payload):
        return {}


class _GraphBadJson:
    async def ainvoke(self, payload):
        return {"messages": [_Message("{not-json")]}


class _GraphOk:
    async def ainvoke(self, payload):
        return {"messages": [_Message('{"answer":"ok"}')]} 


@pytest.mark.asyncio
async def test_generate_chat_completion_raises_upstream_on_missing_messages():
    service = RagService(_GraphMissingMessages(), "sys", "model")
    req = ChatCompletionRequest(model="m", messages=[ChatMessage(role="user", content="hi")])

    with pytest.raises(UpstreamModelError):
        await service.generate_chat_completion(req)


@pytest.mark.asyncio
async def test_generate_chat_completion_raises_upstream_on_bad_json():
    service = RagService(_GraphBadJson(), "sys", "model")
    req = ChatCompletionRequest(model="m", messages=[ChatMessage(role="user", content="hi")])

    with pytest.raises(UpstreamModelError):
        await service.generate_chat_completion(req)


@pytest.mark.asyncio
async def test_generate_chat_completion_succeeds_for_valid_json():
    service = RagService(_GraphOk(), "sys", "model")
    req = ChatCompletionRequest(model="m", messages=[ChatMessage(role="user", content="hi")])

    out = await service.generate_chat_completion(req)
    assert out.resolved_model_id == "model"
    assert out.agent_output["answer"] == "ok"
