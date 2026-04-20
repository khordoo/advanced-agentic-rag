import logging
from langgraph.prebuilt import ToolNode
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from rag_core.genetation.state import RagState
from enum import StrEnum
from rag_core.prompts.loader import AgentPrompt
# from rag_core.schemas.llm import FinalAnswer

from rag_core.core.config import setup_logging
from langfuse import observe

setup_logging()
logger = logging.getLogger(__name__)


class NodeName(StrEnum):
    PRIMARY_AGENT = "primary_agent"
    TOOLS = "tool_execution"
    POST_PROCESSING = "post_processing"
    ROUTER = "router"
    FINISH = "finish"

class GraphNodes:
    def __init__(self, llm: BaseChatModel, tool_executor: ToolNode) -> None:
        self.llm = llm
        self.tool_executor = tool_executor

    @observe(name=NodeName.PRIMARY_AGENT, as_type="llm")
    def execute_primary_agent(self, state: RagState) -> dict:
        logger.info("Executing Agent node...")
        print("Messages: ", state["messages"])

        response = self.llm.invoke(state["messages"])
        logger.info("Agent response received: %s", response)
        return {"messages": [response]}

    @observe(name=NodeName.TOOLS, as_type="tool")
    def execute_tools(self, state: RagState) -> dict:
        logger.info("Executing Tool Execution node...")
        tool_result = self.tool_executor.invoke({"messages": state["messages"]})
        logger.debug("Tool execution results: %s", tool_result)
        return tool_result

    @observe(name=NodeName.POST_PROCESSING, as_type="llm")
    def post_process(self, state: RagState) -> RagState:
        logger.info("Executing Post-Processing node...")
        # Currently a pass-through
        return state

    @observe(name=NodeName.ROUTER, as_type="llm")
    def router(self, state: RagState) -> str:
        last_message = self._get_last_message(state)
        has_tool_calls: bool = bool(getattr(last_message, "tool_calls", None))

        if has_tool_calls:
            logger.info("Router: Detected tool calls. Routing to %s", NodeName.TOOLS)
            return NodeName.TOOLS

        logger.info("Router: No tool calls. Routing to %s", NodeName.FINISH)
        return NodeName.FINISH

    def _get_last_message(self, state: RagState) -> BaseMessage | None:
        messages: list[BaseMessage] = state.get("messages", [])
        return messages[-1] if messages else None
