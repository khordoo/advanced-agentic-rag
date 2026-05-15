import logging

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_rag_service
from app.schemas.openai import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from app.services.rag_service import RagService

openai_router = APIRouter()
logger = logging.getLogger(__name__)


@openai_router.post("/chat/completions", response_model=ChatCompletionResponse)
async def rag(
    chat: ChatCompletionRequest, service: RagService = Depends(get_rag_service)
) -> ChatCompletionResponse:
    if not chat.messages:
        raise HTTPException(status_code=400, detail="Messages array cannot be empty")
    return await service.generate_chat_completion(chat)
