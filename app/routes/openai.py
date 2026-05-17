import logging
import json
import time
import uuid

from fastapi import APIRouter, Depends, Header, Response
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_rag_service
from app.core.config import AppConfig, get_settings
from app.core.exceptions import OpenAIException, UpstreamModelError
from app.schemas.openai import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ModelCard,
    ModelList,
    ResponseFormat,
)
from app.services.formatters import ChatPresenter
from app.services.rag_service import RagService

openai_router = APIRouter()
logger = logging.getLogger(__name__)


def _encode_sse_frame(payload: str) -> bytes:
    return f"data: {payload}\n\n".encode("utf-8")


def _build_streaming_response(model: str, content: str) -> StreamingResponse:
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    first_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    content_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
    }
    final_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }

    async def event_generator():
        yield _encode_sse_frame(json.dumps(first_chunk))
        yield _encode_sse_frame(json.dumps(content_chunk))
        yield _encode_sse_frame(json.dumps(final_chunk))
        yield _encode_sse_frame("[DONE]")

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@openai_router.post("/chat/completions", response_model=ChatCompletionResponse)
async def rag(
    chat: ChatCompletionRequest,
    response: Response,
    x_response_format: ResponseFormat = Header(
        default=ResponseFormat.MARKDOWN,
        alias="x-response-format",
        description="Specify 'markdown' or 'json'",
    ),
    x_request_id: str | None = Header(default=None, alias="x-request-id"),
    service: RagService = Depends(get_rag_service),
) -> ChatCompletionResponse:
    request_id = x_request_id or str(uuid.uuid4())
    response.headers["x-request-id"] = request_id
    start = time.perf_counter()

    if not chat.messages:
        raise OpenAIException(
            status_code=400,
            message="Messages array cannot be empty.",
            error_type="invalid_request_error",
            code="empty_messages",
            param="messages",
        )

    try:
        raw_agent_output = await service.generate_chat_completion(chat)
    except UpstreamModelError:
        raise
    except Exception as exc:
        logger.exception(
            "chat_completion_failed",
            extra={
                "request_id": request_id,
                "requested_model": chat.model,
                "error_class": type(exc).__name__,
            },
        )
        raise OpenAIException(
            status_code=500,
            message="An internal error occurred while generating the completion.",
            error_type="server_error",
            code="internal_server_error",
        ) from exc

    if x_response_format == ResponseFormat.JSON:
        content = ChatPresenter.to_json_string(raw_agent_output.agent_output)
    else:
        content = ChatPresenter.to_markdown(raw_agent_output.agent_output)

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    logger.info(
        "chat_completion_succeeded",
        extra={
            "request_id": request_id,
            "requested_model": chat.model,
            "resolved_model": raw_agent_output.resolved_model_id,
            "response_format": x_response_format,
            "latency_ms": elapsed_ms,
        },
    )

    if chat.stream:
        stream_response = _build_streaming_response(raw_agent_output.resolved_model_id, content)
        stream_response.headers["x-request-id"] = request_id
        return stream_response

    return ChatCompletionResponse(
        model=raw_agent_output.resolved_model_id,
        choices=[ChatChoice(message=ChatMessage(role="assistant", content=content))],
    )


@openai_router.get("/models", response_model=ModelList)
async def list_models(settings: AppConfig = Depends(get_settings)) -> ModelList:
    """Required by Open WebUI and other OpenAI clients to populate model dropdowns."""
    return ModelList(
        data=[
            ModelCard(id=settings.model_id),
        ]
    )
