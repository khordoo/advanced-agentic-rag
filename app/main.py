import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.exceptions import OpenAIException
from app.routes.health import health_router
from app.routes.openai import openai_router
from app.schemas.openai import OpenAIError, OpenAIErrorEnvelope
from app.services.rag_service import RagService
from rag_core.core.telemetry import init_observability
from rag_core.genetation.graph import RagGraphFactory
from rag_core.prompts.loader import AgentPrompt

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    init_observability(service_name="advanced-rag-api")
    graph = RagGraphFactory.build_default()

    agent_prompt = AgentPrompt.from_package("primary_rag_agent")
    static_instructions = agent_prompt.get_system_instructions()
    app.state.rag_service = RagService(
        graph=graph,
        system_instructions=static_instructions,
        model_id=settings.model_id,
    )
    yield


app = FastAPI(lifespan=lifespan)


@app.exception_handler(OpenAIException)
async def openai_exception_handler(request: Request, exc: OpenAIException):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    body = OpenAIErrorEnvelope(
        error=OpenAIError(
            message=exc.message,
            type=exc.error_type,
            param=exc.param,
            code=exc.code,
        )
    )
    headers = {"x-request-id": request_id}
    return JSONResponse(status_code=exc.status_code, content=body.model_dump(), headers=headers)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    logger.warning(
        "request_validation_failed",
        extra={
            "request_id": request_id,
            "path": str(request.url.path),
            "errors": exc.errors(),
        },
    )
    body = OpenAIErrorEnvelope(
        error=OpenAIError(
            message="Request validation failed.",
            type="invalid_request_error",
            param=None,
            code="request_validation_error",
        )
    )
    headers = {"x-request-id": request_id}
    return JSONResponse(status_code=422, content=body.model_dump(), headers=headers)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    logger.exception(
        "unhandled_exception",
        extra={"request_id": request_id, "path": str(request.url.path)},
    )
    body = OpenAIErrorEnvelope(
        error=OpenAIError(
            message="An internal server error occurred.",
            type="server_error",
            param=None,
            code="internal_server_error",
        )
    )
    headers = {"x-request-id": request_id}
    return JSONResponse(status_code=500, content=body.model_dump(), headers=headers)


app.include_router(openai_router, prefix="/v1")
app.include_router(health_router, prefix="/ping", tags=["System"])


@app.get("/")
async def home():
    return {"message": "Hello from RAG. It works!"}
