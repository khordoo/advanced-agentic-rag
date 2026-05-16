import logging

from fastapi import APIRouter, Depends, Header, HTTPException

from app.api.dependencies import get_rag_service
from app.config import AppConfig, get_settings
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


@openai_router.post("/chat/completions", response_model=ChatCompletionResponse)
async def rag(
    chat: ChatCompletionRequest,
    x_response_format: ResponseFormat = Header(
        default=ResponseFormat.MARKDOWN,
        alias="x-response-format",
        description="Specify 'markdown' or 'json'",
    ),
    service: RagService = Depends(get_rag_service),
) -> ChatCompletionResponse:

    if not chat.messages:
        raise HTTPException(status_code=400, detail="Messages array cannot be empty")

    raw_agent_output = await service.generate_chat_completion(chat)
    if x_response_format == ResponseFormat.JSON:
        content = ChatPresenter.to_json_string(raw_agent_output.agent_output)
    else:
        content = ChatPresenter.to_markdown(raw_agent_output.agent_output)

    return ChatCompletionResponse(
        model=raw_agent_output.resolved_model_id,
        choices=[ChatChoice(message=ChatMessage(role="assistant", content=content))],
    )


@openai_router.get("/models", response_model=ModelList)
async def list_models(settings: AppConfig = Depends(get_settings)):
    """
    Required by Open WebUI and other OpenAI clients to populate the model dropdown.
    """
    return ModelList(
        data=[
            ModelCard(id=settings.model_id),
        ]
    )
