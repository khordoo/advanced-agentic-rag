"""Interactive CLI for communicating with the RAG agent graph."""

import logging

from langchain_core.messages import AIMessage, HumanMessage

from rag_core.core.config import setup_logging
from rag_core.core.telemetry import get_observability_callbacks
from rag_core.genetation.graph import RagGraphFactory
from rag_core.genetation.parsers import parse_agent_response
from rag_core.prompts.loader import AgentPrompt

setup_logging()
logger = logging.getLogger(__name__)


def run_chat_loop():
    logger.info("Initializing Agent Prompt & LLM...")
    app = RagGraphFactory.build_default()

    agent_prompt = AgentPrompt.from_package("primary_rag_agent")
    static_instructions = agent_prompt.get_system_instructions()
    messages = []
    dim = "\033[2m"
    reset = "\033[0m"
    print("\n========================================================")
    print("🤖 Interactive RAG Session Started. Type 'exit' to quit.")
    print(f"{dim}Tip: use /new (or /reset) to start a fresh session.{reset}")
    print("========================================================\n")

    langfuse_callbacks = get_observability_callbacks()

    while True:
        try:
            query = input("\nUser > ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        query = query.strip()
        command = query.lower()
        if command in ["exit", "quit", "q", "/q", "/exit"]:
            break
        if not query:
            continue
        if command in ["/new", "/reset"]:
            messages = []
            print("\nStarted a new session.")
            continue

        messages.append(HumanMessage(content=query))

        logger.info("Invoking LangGraph execution...")
        res = app.invoke(
            {"messages": messages, "system_instructions": static_instructions},
            config={
                "callbacks": langfuse_callbacks,
                "metadata": {"env": "local", "app": "cli-langgraph-test"},
            },
        )

        messages = res["messages"]
        response_content = messages[-1].content
        logger.info("Execution complete.")

        try:
            result = parse_agent_response(response_content)
            print("\n----- Agent Response -----")
            print(result.model_dump_json(indent=2))
            messages.append(AIMessage(content=result.model_dump_json()))
        except Exception as e:
            logger.error("Failed to parse response: %s", e)
            logger.error("Raw Agent Response:\n%s", response_content)


if __name__ == "__main__":
    run_chat_loop()
