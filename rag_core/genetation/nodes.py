import json
import logging
import re
from enum import StrEnum

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode
from pydantic import ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from rag_core.core.config import (
    GenerationSettings,
    get_generation_settings,
    setup_logging,
)
from rag_core.genetation.parsers import extract_text_from_message, parse_agent_response
from rag_core.genetation.state import RagState
from rag_core.prompts.loader import AgentPrompt
from rag_core.prompts.packages.intent_router.schema import (
    IntentCategory,
    IntentRouterResponse,
)
from rag_core.prompts.packages.rag_verifier.schema import RagVerifierResponse
from rag_core.prompts.specs.base import RagFinalResponseSchema

setup_logging()
logger = logging.getLogger(__name__)


class NodeName(StrEnum):
    PRIMARY_RAG_AGENT = "primary_rag_agent"
    GENERAL_ASSISTANT_AGENT = "general_assistant_agent"
    SELF_CORRECTING_AGENT = "self_correcting_agent"
    FALLBACK_HANDLER = "fallback_handler"
    INTENT_ROUTER_AGENT = "intent_router_agent"
    INITIAL_ROUTER = "initial_router"
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
        verifier_llm: BaseChatModel,
        fast_llm: BaseChatModel,
        tools: list[BaseTool],
        settings: GenerationSettings | None = None,
    ) -> None:
        self.settings = settings or get_generation_settings()
        self.fast_llm = fast_llm
        self.verifier_llm = verifier_llm
        self.reasoning_llm = reasoning_llm.bind_tools(tools)
        self.tool_executor = ToolNode(tools)

    def route_initial_node(self, state: RagState) -> str:
        if not self.settings.enable_intent_routing:
            logger.info(
                "Initial router selected %s (intent routing disabled).",
                NodeName.PRIMARY_RAG_AGENT,
            )
            return NodeName.PRIMARY_RAG_AGENT

        logger.info(
            "Initial router selected %s (intent routing enabled).",
            NodeName.INTENT_ROUTER_AGENT,
        )
        return NodeName.INTENT_ROUTER_AGENT

    def execute_general_assistant_agent(self, state: RagState) -> dict:
        logger.info("Executing General Assistant Agent node...")
        agent_prompt = AgentPrompt.from_package("general_assistant")
        current_user_query = state["messages"][-self.settings.general_assistant_lookback :]
        prompt = agent_prompt.chat_template.invoke({"question": current_user_query})
        response = self.fast_llm.invoke(prompt.to_messages())
        logger.info("Agent response received: %s", response)
        response_text = extract_text_from_message(response.content)
        formatted = RagFinalResponseSchema(answer=response_text, citations=[])
        return {"messages": [AIMessage(content=formatted.model_dump_json())]}

    def execute_primary_rag_agent(self, state: RagState) -> dict:
        logger.info("Executing Primary RAG Agent node...")
        source_docs = state.get("source_docs", [])
        logger.debug("Source docs count: %d", len(source_docs))
        sys_msg = SystemMessage(content=state["system_instructions"])
        temp_messages = [sys_msg] + state["messages"]
        response = self.reasoning_llm.invoke(temp_messages)
        iteration_count = state.get("iteration_count", 0)
        logger.info("Agent response received. Iteration: %d", iteration_count + 1)
        return {
            "messages": [response],
            "iteration_count": iteration_count + 1,
            "source_docs": source_docs,
        }

    def execute_self_correcting_agent(self, state: RagState) -> dict:
        logger.info("Executing self_correcting node...")
        current_attempts = state.get("self_correction_attempts", 0)

        if not self.settings.enable_grounding_verifier:
            logger.info("Grounding verifier disabled. Skipping verification step.")
            return {
                "verifier_response": RagVerifierResponse(
                    grounded=True,
                    reason="Grounding verifier disabled by configuration.",
                ),
                "self_correction_attempts": current_attempts + 1,
            }

        documents = state.get("source_docs", [])
        if not documents:
            return {
                "verifier_response": RagVerifierResponse(
                    grounded=True, reason="There is no document to verify."
                ),
                "self_correction_attempts": current_attempts + 1,
            }

        generated_answer = extract_text_from_message(state["messages"][-1].content)
        logger.info("Generated answer: %s", generated_answer)
        logger.info("Current self correction attempts: %d", current_attempts)

        agent_prompt = AgentPrompt.from_package("rag_verifier")
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
            response = self.verifier_llm.invoke(prompt.to_messages())
            # This will throw a ValueError if the LLM output is garbage
            response_text = extract_text_from_message(response.content)
            return RagVerifierResponse.model_validate_json(response_text)

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

    def execute_fallback_handler(self, state: RagState) -> dict:
        logger.warning("Circuit breaker tripped. Injecting safe fallback response.")

        # Craft the exact JSON schema your frontend expects, but with a safe failure message
        fallback_response = {
            "answer": "I could not produce a reliable answer based strictly on the provided documents. Please try rephrasing your question or narrowing the scope.",
            "citations": [],
        }

        fallback_msg = AIMessage(content=json.dumps(fallback_response))

        return {"messages": [fallback_msg]}

    def execute_intent_router_agent(self, state: RagState) -> dict:
        """Fast triage agent to classify intent."""
        logger.info("Executing Intent Router node...")
        user_query = state["messages"][-self.settings.intent_router_lookback :]
        agent_prompt = AgentPrompt.from_package("intent_router")
        prompt = agent_prompt.chat_template.invoke(
            {
                "question": user_query,
                "intent_categories": IntentCategory.get_categories(),
            }
        )

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=5),
            retry=retry_if_exception_type((json.decoder.JSONDecodeError, ValidationError)),
            reraise=True,
        )
        def execute_agent(prompt):
            response = self.fast_llm.invoke(prompt.to_messages())
            normalized_content = extract_text_from_message(response.content)
            intent_json = json.loads(normalized_content)
            parsed = IntentRouterResponse.model_validate(intent_json)
            logger.debug("Intent router response: %s", parsed.model_dump_json())
            logger.info("Classified intent: %s", parsed.intent)
            return parsed.intent

        try:
            intent = execute_agent(prompt)
        except Exception as e:
            logger.error("Intent router failed to format even after retries: %s", e)
            intent = IntentCategory.GENERAL_ASSISTANT.value

        return {
            "intent_route": intent,
            "source_docs": [],
            "iteration_count": 0,
            "tool_call_count": 0,
            "self_correction_attempts": 0,
        }

    def route_after_triage(self, state: RagState) -> str:
        """Reads the intent from the state and returns the name of the next node."""
        if IntentCategory.GENERAL_ASSISTANT.value == state.get("intent_route", ""):
            logger.info("Routing to: %s", NodeName.GENERAL_ASSISTANT_AGENT)
            return NodeName.GENERAL_ASSISTANT_AGENT
        logger.info("Routing to: %s", NodeName.PRIMARY_RAG_AGENT)
        return NodeName.PRIMARY_RAG_AGENT

    def execute_tools(self, state: RagState) -> dict:
        logger.info("Executing Tool Execution node...")
        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", None) or []
        logger.debug("Tool calls: %s", tool_calls)

        tool_result = self.tool_executor.invoke({"messages": state["messages"]})

        raw_docs = []
        for msg in tool_result["messages"]:
            if isinstance(msg, ToolMessage):
                try:
                    doc_dicts = json.loads(msg.content)
                    for d in doc_dicts:
                        doc = Document(page_content=d["page_content"], metadata=d["metadata"])
                        raw_docs.append(doc)

                except json.JSONDecodeError:
                    logger.error("Failed to parse tool output as JSON.")
        logger.info("Tool Execution node completed. Extracted %d documents.", len(raw_docs))
        existing_docs = state.get("source_docs", [])
        current_tool_call_count = state.get("tool_call_count", 0)
        return {
            "messages": tool_result["messages"],
            "source_docs": existing_docs + raw_docs,
            "tool_call_count": current_tool_call_count + len(tool_calls),
        }

    @staticmethod
    def _get_doc_source(doc: Document) -> str:
        """Return the best available source reference from document metadata."""
        metadata = doc.metadata or {}
        return (
            metadata.get("source_url")
            or metadata.get("source")
            or metadata.get("source_path")
            or ""
        )

    @staticmethod
    def _find_source_doc(
        citation_asset_key: str,
        docs_lookup: dict[str, Document],
    ) -> Document | None:
        """Find the source document matching a citation's exact asset_key."""
        return docs_lookup.get(citation_asset_key)

    @staticmethod
    def _dedupe_and_remap_citations(parsed_response: RagFinalResponseSchema) -> dict[int, int]:
        """Deduplicate citations by (asset_key, page_number) and remap indexes."""
        deduped = []
        seen_by_key: dict[tuple[str, int], int] = {}
        old_to_new: dict[int, int] = {}
        next_index = 1

        for citation in parsed_response.citations:
            old_index = citation.document_index
            key = (citation.asset_key, citation.page_number)
            existing = seen_by_key.get(key)
            if existing is None:
                new_index = next_index
                next_index += 1
                seen_by_key[key] = new_index
                citation.document_index = new_index
                deduped.append(citation)
            else:
                new_index = existing
            old_to_new[old_index] = new_index

        parsed_response.citations = deduped
        return old_to_new

    @staticmethod
    def _remap_inline_citation_indexes(answer: str, index_map: dict[int, int]) -> str:
        """Rewrite exact [N] citations using safe regex replacement."""
        if not index_map:
            return answer

        def _replace(match: re.Match[str]) -> str:
            old_idx = int(match.group(1))
            new_idx = index_map.get(old_idx)
            if new_idx is None:
                return match.group(0)
            return f"[{new_idx}]"

        remapped = re.sub(r"\[(\d+)\]", _replace, answer)
        # Collapse adjacent duplicates like [1][1], [1] [1], or [1], [1].
        remapped = re.sub(r"(\[(\d+)\])(?:[\s,]*\[\2\])+", r"\1", remapped)
        return remapped

    def execute_post_processing(self, state: RagState) -> dict:
        logger.info("Executing Post-Processing node...")
        try:
            raw_content = extract_text_from_message(state["messages"][-1].content)
            parsed_response = parse_agent_response(raw_content)

            # Build lookup from actual retrieved documents keyed only by asset_key.
            docs = state.get("source_docs", [])
            docs_lookup: dict[str, Document] = {}
            for doc in docs:
                asset_key = doc.metadata.get("asset_key", "")
                if asset_key:
                    docs_lookup[asset_key] = doc

            for citation in parsed_response.citations:
                doc = self._find_source_doc(citation.asset_key, docs_lookup)
                if doc:
                    citation.asset_name = (
                        doc.metadata.get("asset_name") or doc.metadata.get("file_name") or ""
                    )
                    citation.source_url = self._get_doc_source(doc)

            citation_index_map = self._dedupe_and_remap_citations(parsed_response)
            parsed_response.answer = self._remap_inline_citation_indexes(
                parsed_response.answer,
                citation_index_map,
            )

            print(parsed_response.answer)
            logger.info("Post-Processing node completed. Parsed response: %s", parsed_response)
            return {"messages": [AIMessage(content=parsed_response.model_dump_json())]}
        except Exception as e:
            logger.error("Post-processing failed, returning raw response: %s", e)
            raw_content = (
                extract_text_from_message(state["messages"][-1].content)
                if state.get("messages")
                else ""
            )
            fallback = RagFinalResponseSchema(answer=raw_content, citations=[])
            return {"messages": [AIMessage(content=fallback.model_dump_json())]}

    def tool_router(self, state: RagState) -> str:
        iteration_count = state.get("iteration_count", 0)
        logger.debug("Global iteration count: %d", iteration_count)
        if iteration_count >= self.settings.max_agent_iterations:
            logger.info(
                "Max agent iterations reached: %d. Routing to FALLBACK_HANDLER.",
                iteration_count,
            )
            return NodeName.FALLBACK_HANDLER

        last_message = self._get_last_message(state)
        has_tool_calls = bool(getattr(last_message, "tool_calls", None))

        if not has_tool_calls:
            logger.info("Router: No tool calls. Routing to %s", NodeName.FINISH)
            return NodeName.FINISH

        tool_call_count = state.get("tool_call_count", 0)
        logger.debug("Tool call count: %d", tool_call_count)
        if tool_call_count >= self.settings.max_tool_calls:
            logger.info(
                "Max tool calls reached: %d. Routing to FALLBACK_HANDLER.",
                tool_call_count,
            )
            return NodeName.FALLBACK_HANDLER

        logger.info("Router: Detected tool calls. Routing to %s", NodeName.TOOLS)
        return NodeName.TOOLS

    def verifier_router(self, state: RagState) -> str:
        verifier_resp = state.get("verifier_response")
        if not verifier_resp:
            logger.debug("No verifier response in state. Routing to FINISH.")
            return NodeName.FINISH

        logger.debug("Verifier response — grounded: %s", verifier_resp.grounded)
        if verifier_resp.grounded:
            return NodeName.FINISH

        current_attempts = state.get("self_correction_attempts", 0)
        if current_attempts >= self.settings.max_self_correction_attempts:
            logger.warning(
                "Max verifier attempts reached: %d. Routing to FALLBACK_HANDLER.",
                current_attempts,
            )
            return NodeName.FALLBACK_HANDLER
        else:
            logger.info(
                "Verifier attempts: %d/%d. Routing to %s.",
                current_attempts,
                self.settings.max_self_correction_attempts,
                NodeName.PRIMARY_RAG_AGENT,
            )
            return NodeName.PRIMARY_RAG_AGENT

    def _get_last_message(self, state: RagState) -> BaseMessage | None:
        messages: list[BaseMessage] = state.get("messages", [])
        return messages[-1] if messages else None
