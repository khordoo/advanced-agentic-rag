from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.routes.health import health_router
from app.routes.openai import openai_router
from app.services.rag_service import RagService
from rag_core.genetation.graph import RagGraphFactory
from rag_core.prompts.loader import AgentPrompt


@asynccontextmanager
async def lifespan(app: FastAPI):
    graph = RagGraphFactory.build_default()

    agent_prompt = AgentPrompt.from_package("primary_rag_agent")
    static_instructions = agent_prompt.get_system_instructions()
    app.state.rag_service = RagService(graph=graph, system_instructions=static_instructions)
    yield


app = FastAPI(lifespan=lifespan)

app.include_router(openai_router, prefix="/v1")
app.include_router(health_router, prefix="/health")


@app.get("/")
async def home():
    return {"message": "Hello from RAG. It works!"}
