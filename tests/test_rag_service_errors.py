import pytest

from app.core.exceptions import UpstreamModelError
from app.schemas.openai import ChatCompletionRequest, ChatMessage
from app.services.rag_service import RagService, _IncrementalAnswerJSONParser
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


class _GraphStreamOk:
    async def astream_events(self, payload, version):
        assert version == "v2"
        chunks = ['{"answer":"Hel', 'lo world","citations":[{"asset_key":"doc","page_number":1}]}']
        for c in chunks:
            yield {"event": "on_chat_model_stream", "data": {"chunk": _Message(c)}}


class _GraphStreamBadJson:
    async def astream_events(self, payload, version):
        yield {"event": "on_chat_model_stream", "data": {"chunk": _Message('{"answer":"x"')}}


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


def test_incremental_answer_parser_handles_escaped_quotes():
    parser = _IncrementalAnswerJSONParser()
    out1 = parser.feed('{"answer":"Hello \\"')
    out2 = parser.feed('world\\" there","citations":[]}')
    assert out1 == 'Hello "'
    assert out2 == 'world" there'


@pytest.mark.asyncio
async def test_stream_chat_completion_extracts_answer_and_sources():
    service = RagService(_GraphStreamOk(), "sys", "model")
    req = ChatCompletionRequest(model="m", messages=[ChatMessage(role="user", content="hi")])

    chunks = []
    async for piece in service.stream_chat_completion(req, response_format="markdown"):
        chunks.append(piece)

    output = "".join(chunks)
    assert "Hello world" in output
    assert "**Sources:**" in output


@pytest.mark.asyncio
async def test_stream_chat_completion_emits_graceful_error_on_malformed_json_after_emitting():
    service = RagService(_GraphStreamBadJson(), "sys", "model")
    req = ChatCompletionRequest(model="m", messages=[ChatMessage(role="user", content="hi")])

    chunks = []
    async for piece in service.stream_chat_completion(req, response_format="markdown"):
        chunks.append(piece)
    assert any("System Error" in chunk for chunk in chunks)


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
