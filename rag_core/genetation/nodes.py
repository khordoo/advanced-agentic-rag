import logging
import json
from langgraph.prebuilt import ToolNode
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.documents import Document
from rag_core.genetation.state import RagState
from enum import StrEnum
from rag_core.prompts.loader import AgentPrompt
from rag_core.prompts.specs.base import RagFinalResponseSchema
from rag_core.genetation.parsers import extract_clean_json

# from rag_core.schemas.llm import FinalAnswer
from langfuse import observe
from rag_core.core.config import setup_logging
from rag_core.core.telemetry import init_langfuse_client

setup_logging()
logger = logging.getLogger(__name__)

# Initialise the Langfuse SDK singleton so @observe decorators send traces
init_langfuse_client()


class NodeName(StrEnum):
    PRIMARY_RAG_AGENT = "primary_rag_agent"
    GENERAL_ASSISTANT_AGENT = "general_assistant_agent"
    VERIFIER_AGENT = "verifier_agent"
    INTENT_ROUTER_AGENT = "intent_router_agent"
    TOOLS = "tool_execution"
    POST_PROCESSING = "post_processing"
    TOOL_ROUTER = "tool_router"
    INTENT_ROUTER = "intent_router"
    TRIAGE_ROUTER = "triage_router"
    FINISH = "finish"


class GraphNodes:
    def __init__(
        self,
        reasoning_llm: BaseChatModel,
        fast_llm: BaseChatModel,
        tools: list[BaseTool],
    ) -> None:
        self.fast_llm = fast_llm
        self.reasoning_llm = reasoning_llm.bind_tools(tools)
        self.tool_executor = ToolNode(tools)

    @observe(name=NodeName.GENERAL_ASSISTANT_AGENT, as_type="agent")
    def execute_general_assistant_agent(self, state: RagState) -> dict:
        logger.info("Executing Agent node...")
        agent_prompt = AgentPrompt.from_yaml("general_assistant")
        current_user_query = state["messages"][-1].content
        prompt = agent_prompt.chat_template.invoke({"question": current_user_query})
        response = self.fast_llm.invoke(prompt.to_messages())
        logger.info("Agent response received: %s", response)
        formatted = RagFinalResponseSchema(answer=response.content, citations=[])
        return {"messages": [AIMessage(content=formatted.model_dump_json())]}

    @observe(name=NodeName.PRIMARY_RAG_AGENT, as_type="agent")
    def execute_primary_rag_agent(self, state: RagState) -> dict:
        logger.info("Executing Agent node...")
        print("--------------Len of source docs", len(state["source_docs"]))
        response = self.reasoning_llm.invoke(state["messages"])
        logger.info("Agent response received.")
        return {"messages": [response]}

    @observe(name=NodeName.VERIFIER_AGENT, as_type="evaluator")
    def execute_verifier(self, state: RagState) -> dict:
        logger.info("Executing Verifier node...")

        documents = state.get("context", "")
        # Safely extract the generated answer string
        last_msg = state["messages"][-1].content
        generated_answer = extract_clean_json(last_msg).get("answer", "")

        agent_prompt = AgentPrompt.from_yaml("rag_verifier")
        prompt = agent_prompt.chat_template.invoke(
            {"documents": documents, "answer": generated_answer}
        )

        # Use the fast_llm to grade
        response = self.fast_llm.invoke(prompt.to_messages())

        # Parse the Verifier's JSON response
        try:
            grade_dict = extract_clean_json(response.content)
            is_grounded = grade_dict.get("grounded", False)
            reason = grade_dict.get("reason", "")
        except ValueError:
            # If the verifier fails to format, default to safe (not grounded)
            is_grounded = False
            reason = "The verifier could not validate the formatting. Please ensure strict adherence to the facts."

        if is_grounded:
            logger.info("Answer verified as grounded.")
            return {"grounded": True}
        else:
            logger.warning(f"Hallucination detected. Reason: {reason}")

            # THE MAGIC: Create a feedback message to push back to the agent
            feedback_msg = AIMessage(
                content=f"Your previous answer was rejected by the verifier. Reason: {reason}. Please rewrite your final response using ONLY the provided documents."
            )

            # By returning the feedback in the messages array, the primary agent
            # will read it on its next execution loop!
            return {"grounded": False, "messages": [feedback_msg]}

    @observe(name=NodeName.INTENT_ROUTER_AGENT, as_type="agent")
    def execute_intent_router_agent(self, state: RagState) -> dict:
        """Fast triage agent to classify intent."""
        # Grab the user's latest message
        user_query = state["messages"][-1].content
        agent_prompt = AgentPrompt.from_yaml("intent_router")
        prompt = agent_prompt.chat_template.invoke({"question": user_query})
        response = self.fast_llm.invoke(prompt.to_messages())
        intent = response.content.strip().upper()
        # Save the route decision to the state so the graph knows where to go
        return {
            "route": AIMessage(content=intent),
            "source_docs": [],  # clean slate for a new query
        }

    @observe(name=NodeName.INTENT_ROUTER, as_type="evaluator")
    def route_after_triage(self, state: RagState) -> str:
        """Reads the intent from the state and returns the name of the next node."""
        if "GENERAL_ASSISTANT" in state.get("route", ""):
            print("-----------Routing to : ", NodeName.GENERAL_ASSISTANT_AGENT)
            return NodeName.GENERAL_ASSISTANT_AGENT
        print("----------------routing to : ", NodeName.PRIMARY_RAG_AGENT)
        # Default to your heavy reasoning agent if it's not explicitly chitchat
        return NodeName.PRIMARY_RAG_AGENT

    @observe(name=NodeName.TOOLS, as_type="tool")
    def execute_tools(self, state: RagState) -> dict:
        print("-------------------Executing Tool Execution node-------------------")
        print("tool calls", state["messages"][-1].tool_calls)
        logger.info("Executing Tool Execution node...")

        tool_result = self.tool_executor.invoke({"messages": state["messages"]})

        raw_docs = []
        for msg in tool_result["messages"]:
            if isinstance(msg, ToolMessage):
                try:
                    doc_dicts = json.loads(msg.content)
                    for d in doc_dicts:
                        doc = Document(
                            page_content=d["page_content"], metadata=d["metadata"]
                        )
                        raw_docs.append(doc)

                except json.JSONDecodeError:
                    logger.error("Failed to parse tool output as JSON.")
        print("-------------------Tool Execution node Completed-------------------")
        existing_docs = state.get("source_docs", [])
        return {
            "messages": tool_result["messages"],
            "source_docs": existing_docs + raw_docs,  # Keep for verifier
        }

    @observe(name=NodeName.POST_PROCESSING, as_type="tool")
    def post_process(self, state: RagState) -> RagState:
        logger.info("Executing Post-Processing node...")
        # Currently a pass-through
        return state

    @observe(name=NodeName.TOOL_ROUTER, as_type="evaluator")
    def tool_router(self, state: RagState) -> str:
        last_message = self._get_last_message(state)
        has_tool_calls: bool = bool(getattr(last_message, "tool_calls", None))

        if has_tool_calls:
            logger.info("Router: Detected tool calls. Routing to %s", NodeName.TOOLS)
            return NodeName.TOOLS

        logger.info("Router: No tool calls. Routing to %s", NodeName.FINISH)
        return NodeName.FINISH

    def verifier_router(self, state: RagState) -> str:
        if state.get("grounded", False):
            return NodeName.PRIMARY_RAG_AGENT
        else:
            return NodeName.FINISH

    def _get_last_message(self, state: RagState) -> BaseMessage | None:
        messages: list[BaseMessage] = state.get("messages", [])
        return messages[-1] if messages else None
