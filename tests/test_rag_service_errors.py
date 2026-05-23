import pytest

from app.core.exceptions import UpstreamModelError
from app.schemas.openai import ChatCompletionRequest, ChatMessage
from app.services.rag_service import RagService
from rag_core.core.readiness import SetupRequiredError


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


class _SetupRequiredChecker:
    def ensure_ready(self) -> None:
        raise SetupRequiredError(
            reason="schema_not_migrated",
            message="The database schema is not ready. Run `make ingest`.",
        )


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


@pytest.mark.asyncio
async def test_generate_chat_completion_checks_readiness_before_graph_invocation():
    service = RagService(
        _GraphOk(),
        "sys",
        "model",
        readiness_checker=_SetupRequiredChecker(),
    )
    req = ChatCompletionRequest(model="m", messages=[ChatMessage(role="user", content="hi")])

    with pytest.raises(SetupRequiredError) as exc_info:
        await service.generate_chat_completion(req)

    assert exc_info.value.reason == "schema_not_migrated"


def test_create_event_stream_checks_readiness_before_graph_invocation():
    service = RagService(
        _GraphOk(),
        "sys",
        "model",
        readiness_checker=_SetupRequiredChecker(),
    )
    req = ChatCompletionRequest(model="m", messages=[ChatMessage(role="user", content="hi")])

    with pytest.raises(SetupRequiredError) as exc_info:
        service.create_event_stream(req)

    assert exc_info.value.reason == "schema_not_migrated"
