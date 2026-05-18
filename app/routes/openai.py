import logging
import asyncio
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


def _chunk_text(content: str, chunk_size: int = 160) -> list[str]:
    if chunk_size <= 0:
        return [content]
    return [content[i : i + chunk_size] for i in range(0, len(content), chunk_size)] or [""]


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

    if chat.stream:
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        generation_task = asyncio.create_task(service.generate_chat_completion(chat))

        async def stream_events():
            role_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": service.model_id,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            yield _encode_sse_frame(json.dumps(role_chunk))
            status_sent = False
            delayed_status_sent = False
            loop_start = time.perf_counter()

            if x_response_format == ResponseFormat.MARKDOWN:
                status_sent = True
                yield _encode_sse_frame(
                    json.dumps(
                        {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": service.model_id,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "content": "<think>\nAnalyzing request and routing to the right agent...\n"
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
                )

            while not generation_task.done():
                await asyncio.sleep(0.2)
                if (
                    x_response_format == ResponseFormat.MARKDOWN
                    and not delayed_status_sent
                    and time.perf_counter() - loop_start >= 2.0
                ):
                    delayed_status_sent = True
                    yield _encode_sse_frame(
                        json.dumps(
                            {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": service.model_id,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {
                                            "content": "Retrieving and verifying supporting documents...\n"
                                        },
                                        "finish_reason": None,
                                    }
                                ],
                            }
                        )
                    )

            try:
                raw_agent_output = await generation_task
            except Exception:
                if status_sent:
                    closing_think_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": service.model_id,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": "</think>\n"},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield _encode_sse_frame(json.dumps(closing_think_chunk))
                error_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": service.model_id,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "\n\n*[System Error: completion failed.]*"},
                            "finish_reason": None,
                        }
                    ],
                }
                final_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": service.model_id,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield _encode_sse_frame(json.dumps(error_chunk))
                yield _encode_sse_frame(json.dumps(final_chunk))
                yield _encode_sse_frame("[DONE]")
                return

            if x_response_format == ResponseFormat.JSON:
                content = ChatPresenter.to_json_string(raw_agent_output.agent_output)
            else:
                if status_sent:
                    closing_think_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": raw_agent_output.resolved_model_id,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": "</think>\n\n"},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield _encode_sse_frame(json.dumps(closing_think_chunk))
                content = ChatPresenter.to_markdown(raw_agent_output.agent_output)

            for delta in _chunk_text(content):
                content_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": raw_agent_output.resolved_model_id,
                    "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                }
                yield _encode_sse_frame(json.dumps(content_chunk))

            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": raw_agent_output.resolved_model_id,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield _encode_sse_frame(json.dumps(final_chunk))
            yield _encode_sse_frame("[DONE]")

        stream_response = StreamingResponse(stream_events(), media_type="text/event-stream")
        stream_response.headers["x-request-id"] = request_id
        return stream_response

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
