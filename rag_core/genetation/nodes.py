import logging
import json
from langgraph.prebuilt import ToolNode
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    ToolMessage,
    HumanMessage,
)
from langchain_core.documents import Document
from rag_core.genetation.state import RagState
from rag_core.prompts.specs.rag_verifier import RagVerifierResponse
from enum import StrEnum
from rag_core.prompts.loader import AgentPrompt
from rag_core.prompts.specs.base import RagFinalResponseSchema
from rag_core.genetation.parsers import extract_clean_json
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)


from langfuse import observe
from rag_core.core.config import setup_logging, get_generation_settings
from rag_core.core.telemetry import init_langfuse_client

setup_logging()
logger = logging.getLogger(__name__)

# Initialise the Langfuse SDK singleton so @observe decorators send traces
init_langfuse_client()
settings = get_generation_settings()


class NodeName(StrEnum):
    PRIMARY_RAG_AGENT = "primary_rag_agent"
    GENERAL_ASSISTANT_AGENT = "general_assistant_agent"
    SELF_CORRECTING_AGENT = "self_correcting_agent"
    FALLBACK_HANDLER = "fallback_handler"
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
        max_iterations: int = None,
        max_self_correction_attempts: int = None,
    ) -> None:
        self.fast_llm = fast_llm
        self.reasoning_llm = reasoning_llm.bind_tools(tools)
        self.tool_executor = ToolNode(tools)
        self.max_iterations = max_iterations or settings.max_agent_iterations
        self.max_self_correction_attempts = (
            max_self_correction_attempts or settings.max_self_correction_attempts
        )

    @observe(name=NodeName.GENERAL_ASSISTANT_AGENT, as_type="agent")
    def execute_general_assistant_agent(self, state: RagState) -> dict:
        logger.info("Executing General Assistant Agent node...")
        agent_prompt = AgentPrompt.from_yaml("general_assistant")
        current_user_query = state["messages"][-1].content
        prompt = agent_prompt.chat_template.invoke({"question": current_user_query})
        response = self.fast_llm.invoke(prompt.to_messages())
        logger.info("Agent response received: %s", response)
        formatted = RagFinalResponseSchema(answer=response.content, citations=[])
        return {"messages": [AIMessage(content=formatted.model_dump_json())]}

    @observe(name=NodeName.PRIMARY_RAG_AGENT, as_type="agent")
    def execute_primary_rag_agent(self, state: RagState) -> dict:
        logger.info("Executing Primary RAG Agent node...")
        logger.debug("Source docs count: %d", len(state["source_docs"]))
        response = self.reasoning_llm.invoke(state["messages"])
        iteration_count = state.get("iteration_count", 0)
        logger.info("Agent response received. Iteration: %d", iteration_count)
        return {"messages": [response], "iteration_count": iteration_count + 1}

    @observe(name=NodeName.SELF_CORRECTING_AGENT, as_type="evaluator")
    def execute_self_correcting_agent(self, state: RagState) -> dict:
        logger.info("Executing self_correcting node...")

        documents = state.get("source_docs", [])
        generated_answer = state["messages"][-1].content
        current_attempts = state.get("self_correction_attempts", 0)

        agent_prompt = AgentPrompt.from_yaml("rag_verifier")
        prompt = agent_prompt.chat_template.invoke(
            {"documents": documents, "answer": generated_answer}
        )

        # --- TENACITY RETRY BLOCK ---
        # Retry up to 3 times, waiting a bit between each attempt.
        # Only retries if Pydantic throws a ValueError (JSON/Validation failure).
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=5),
            retry=retry_if_exception_type(ValueError),
            reraise=True,  # If it fails 3 times, pass the error down to the try/except block
        )
        def fetch_and_parse_grade():
            logger.info("Invoking Verifier LLM...")
            response = self.fast_llm.invoke(prompt.to_messages())
            # This will throw a ValueError if the LLM output is garbage
            return RagVerifierResponse.model_validate_json(response.content)

        # --- PARSING & FALLBACK ---
        try:
            result = fetch_and_parse_grade()
            is_grounded = result.grounded
            reason = result.reason
            logger.info("Verifier response parsed successfully: %s", result)

        except ValueError as e:
            logger.error("Verifier failed to format even after retries: %s", e)
            is_grounded = False
            reason = "The verifier could not validate the formatting. Please ensure strict adherence to the facts."

        # --- ROUTING & STATE UPDATE ---
        if is_grounded:
            logger.info("Answer verified as grounded.")
            return {
                "verifier_response": RagVerifierResponse(grounded=True, reason=reason),
                "self_correction_attempts": current_attempts + 1,
            }
        else:
            logger.warning("Hallucination detected. Reason: %s", reason)
            feedback_msg = (
                f"Your previous answer was rejected by the verifier. "
                f"Reason: {reason}. Please rewrite your final response using ONLY the provided documents."
            )

            return {
                "verifier_response": RagVerifierResponse(
                    grounded=False,
                    reason=feedback_msg,
                ),
                "self_correction_attempts": current_attempts + 1,
                "messages": [HumanMessage(content=feedback_msg)],
            }

    @observe(name=NodeName.FALLBACK_HANDLER, as_type="generation")
    def execute_fallback_handler(self, state: RagState) -> dict:
        logger.warning("Circuit breaker tripped. Injecting safe fallback response.")

        # Craft the exact JSON schema your frontend expects, but with a safe failure message
        fallback_response = {
            "answer": "I could not produce a reliable answer based strictly on the provided documents. Please try rephrasing your question or narrowing the scope.",
            "citations": [],
        }

        fallback_msg = AIMessage(content=json.dumps(fallback_response))

        return {"messages": [fallback_msg]}

    @observe(name=NodeName.INTENT_ROUTER_AGENT, as_type="agent")
    def execute_intent_router_agent(self, state: RagState) -> dict:
        """Fast triage agent to classify intent."""
        logger.info("Executing Intent Router node...")
        user_query = state["messages"][-1].content
        agent_prompt = AgentPrompt.from_yaml("intent_router")
        prompt = agent_prompt.chat_template.invoke({"question": user_query})
        response = self.fast_llm.invoke(prompt.to_messages())
        intent = response.content.strip().upper()
        logger.info("Classified intent: %s", intent)
        return {
            "intent_route": intent,
            "source_docs": [],
            "iteration_count": 0,
            "self_correction_attempts": 0,
        }

    @observe(name=NodeName.INTENT_ROUTER, as_type="evaluator")
    def route_after_triage(self, state: RagState) -> str:
        """Reads the intent from the state and returns the name of the next node."""
        if "GENERAL_ASSISTANT" in state.get("intent_route", ""):
            logger.info("Routing to: %s", NodeName.GENERAL_ASSISTANT_AGENT)
            return NodeName.GENERAL_ASSISTANT_AGENT
        logger.info("Routing to: %s", NodeName.PRIMARY_RAG_AGENT)
        return NodeName.PRIMARY_RAG_AGENT

    @observe(name=NodeName.TOOLS, as_type="tool")
    def execute_tools(self, state: RagState) -> dict:
        logger.info("Executing Tool Execution node...")
        logger.debug("Tool calls: %s", state["messages"][-1].tool_calls)

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
        logger.info("Tool Execution node completed. Extracted %d documents.", len(raw_docs))
        existing_docs = state.get("source_docs", [])
        return {
            "messages": tool_result["messages"],
            "source_docs": existing_docs + raw_docs,
        }

    @observe(name=NodeName.POST_PROCESSING, as_type="tool")
    def post_process(self, state: RagState) -> RagState:
        logger.info("Executing Post-Processing node...")
        # Currently a pass-through
        return state

    @observe(name=NodeName.TOOL_ROUTER, as_type="evaluator")
    def tool_router(self, state: RagState) -> str:
        iteration_count = state.get("iteration_count", 0)
        logger.debug("Tool iteration count: %d", iteration_count)
        if iteration_count >= self.max_iterations:
            logger.info(
                "Max iterations reached: %d. Routing to FALLBACK_HANDLER.",
                iteration_count,
            )
            return NodeName.FALLBACK_HANDLER

        last_message = self._get_last_message(state)
        has_tool_calls = bool(getattr(last_message, "tool_calls", None))

        if has_tool_calls:
            logger.info("Router: Detected tool calls. Routing to %s", NodeName.TOOLS)
            return NodeName.TOOLS

        logger.info("Router: No tool calls. Routing to %s", NodeName.FINISH)
        return NodeName.FINISH

    def verifier_router(self, state: RagState) -> str:
        verifier_resp = state.get("verifier_response")
        if not verifier_resp:
            logger.debug("No verifier response in state. Routing to FINISH.")
            return NodeName.FINISH

        logger.debug("Verifier response — grounded: %s", verifier_resp.grounded)
        if verifier_resp.grounded:
            return NodeName.FINISH

        current_attempts = state.get("self_correction_attempts", 0)
        if current_attempts >= self.max_self_correction_attempts:
            logger.warning(
                "Max verifier attempts reached: %d. Routing to FALLBACK_HANDLER.",
                current_attempts,
            )
            return NodeName.FALLBACK_HANDLER
        else:
            logger.info(
                "Verifier attempts: %d/%d. Routing to %s.",
                current_attempts,
                self.max_self_correction_attempts,
                NodeName.PRIMARY_RAG_AGENT,
            )
            return NodeName.PRIMARY_RAG_AGENT

    def _get_last_message(self, state: RagState) -> BaseMessage | None:
        messages: list[BaseMessage] = state.get("messages", [])
        return messages[-1] if messages else None
