"""FastAPI server for the RAG agent web UI."""

import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from langchain_core.messages import HumanMessage, AIMessage

from rag_core.agents import AgentFactory
from rag_core.core.config import setup_logging
from rag_core.core.telemetry import get_observability_callbacks
from rag_core.genetation.graph import RagGraphFactory
from rag_core.prompts.loader import AgentPrompt
from rag_core.prompts.specs.base import RagFinalResponseSchema

setup_logging()
logger = logging.getLogger(__name__)

# ── In-memory session store ──────────────────────────────────────────────
# Maps session_id → list[BaseMessage]
_sessions: dict[str, list] = {}

# ── Shared singletons (populated in lifespan) ───────────────────────────
_app_graph = None
_agent_prompt = None
_langfuse_callbacks = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the LangGraph app & prompt once at startup."""
    global _app_graph, _agent_prompt, _langfuse_callbacks

    logger.info("🚀  Initializing RAG Web Server...")
    _agent_prompt = AgentPrompt.from_package("primary_rag_agent")
    reasoning_llm = AgentFactory.create(
        provider="aws", model_name="amazon.nova-pro-v1:0"
    )
    fast_llm = AgentFactory.create(provider="aws", model_name="amazon.nova-micro-v1:0")
    _app_graph = RagGraphFactory.build(reasoning_llm=reasoning_llm, fast_llm=fast_llm)
    _langfuse_callbacks = get_observability_callbacks()
    logger.info("✅  RAG graph ready.")

    yield  # ── app is running ──

    logger.info("🛑  Shutting down RAG Web Server.")


# ── FastAPI app ──────────────────────────────────────────────────────────
app = FastAPI(title="Advanced RAG Chat", lifespan=lifespan)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Routes ───────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the chat UI."""
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/api/chat")
async def chat(request: Request):
    """Handle a chat message and return a streaming SSE response."""
    body = await request.json()
    query: str = body.get("message", "").strip()
    session_id: str = body.get("session_id") or str(uuid.uuid4())

    if not query:
        return {"error": "Empty message"}

    messages = _sessions.get(session_id, [])

    # Build or append messages
    if not messages:
        prompt = _agent_prompt.chat_template.invoke({"question": query})
        messages = prompt.to_messages()
    else:
        messages.append(HumanMessage(content=query))

    async def _generate():
        nonlocal messages
        yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"
        yield f"data: {json.dumps({'type': 'status', 'content': 'Thinking...'})}\n\n"

        try:
            logger.info("Invoking LangGraph for session=%s", session_id)
            res = _app_graph.invoke(
                {"messages": messages},
                config={
                    "callbacks": _langfuse_callbacks,
                    "metadata": {"env": "local", "app": "web-ui"},
                },
            )

            messages = res["messages"]
            response_content = messages[-1].content
            logger.info("Graph execution complete for session=%s", session_id)

            # Post-processing already returns RagFinalResponseSchema JSON
            try:
                result = RagFinalResponseSchema.model_validate_json(response_content)
                answer = result.answer
                citations = [c.model_dump() for c in result.citations]
                messages.append(AIMessage(content=response_content))
            except Exception:
                answer = response_content
                citations = []
                messages.append(AIMessage(content=response_content))

            _sessions[session_id] = messages

            yield f"data: {json.dumps({'type': 'answer', 'content': answer, 'citations': citations})}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.exception("Error during graph invocation")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")


@app.post("/api/reset")
async def reset_session(request: Request):
    """Clear a chat session."""
    body = await request.json()
    session_id = body.get("session_id")
    if session_id and session_id in _sessions:
        del _sessions[session_id]
    return {"status": "ok"}


# ── Entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("rag_core.web.server:app", host="0.0.0.0", port=8000, reload=True)
