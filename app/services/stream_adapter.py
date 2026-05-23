"""
Token Adapter: Translates LangGraph's internal event firehose into
two clean SSE channels for Open WebUI.

Channel 1: <think> tags — live status updates from intermediate graph nodes.
Channel 2: content     — buffered final answer, yielded after post-processing.

Design constraints:
    - Single contiguous <think> block (Open WebUI renders this best).
    - Final answer is ALWAYS buffered so post-processing can remap citation
      indexes before the client sees them.
    - Node names are referenced via the ``NodeName`` enum so future renames
      propagate automatically.
"""

import json
import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum

from app.services.formatters import ChatPresenter
from rag_core.genetation.nodes import NodeName

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------


class StreamPhase(StrEnum):
    """Logical phases visible to the end-user in the <think> block."""

    ROUTING = "routing"
    RETRIEVAL = "retrieval"
    REASONING = "reasoning"
    VERIFICATION = "verification"
    POST_PROCESSING = "post_processing"
    COMPLETE = "complete"


# Maps LangGraph node names → (phase, human-readable status message).
# Keys use the ``NodeName`` enum so a rename in rag_core propagates here.
_NODE_STATUS_MAP: dict[str, tuple[StreamPhase, str]] = {
    NodeName.INTENT_ROUTER_AGENT: (
        StreamPhase.ROUTING,
        "Evaluating search intent and routing...", 
    ),
    NodeName.PRIMARY_RAG_AGENT: (
        StreamPhase.REASONING,
        "Reasoning over retrieved documents...", 
    ),
    NodeName.TOOLS: (
        StreamPhase.RETRIEVAL,
        "Searching knowledge base...",
    ),
    NodeName.SELF_CORRECTING_AGENT: (
        StreamPhase.VERIFICATION,
        "Running factual consistency checks...",
    ),
    NodeName.POST_PROCESSING: (
        StreamPhase.POST_PROCESSING,
        "Standardizing output and citations...", 
    ),
    NodeName.FALLBACK_HANDLER: (
        StreamPhase.COMPLETE,
        "Formulating fallback response (no exact matches found)...",
    ),
    NodeName.GENERAL_ASSISTANT_AGENT: (
        StreamPhase.REASONING,
        "Drafting general conversational response...", 
    ),
}
# Nodes whose ``on_chain_end`` carries the final ``AIMessage`` JSON.
_TERMINAL_NODES: frozenset[str] = frozenset(
    {
        NodeName.POST_PROCESSING,
        NodeName.GENERAL_ASSISTANT_AGENT,
        NodeName.FALLBACK_HANDLER,
    }
)


@dataclass
class _StreamState:
    """Mutable bookkeeping for the adapter coroutine."""

    phase: StreamPhase = StreamPhase.ROUTING
    phases_emitted: set[str] = field(default_factory=set)
    think_open: bool = False
    final_json: dict | None = None
    raw_fallback: str = ""
    retry_count: int = 0
    last_intent_line: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_message_content(messages: list) -> str | None:
    """Pull the ``content`` string from the last message in a state update."""
    if not messages:
        return None
    last = messages[-1]
    # Handle both LangChain objects and plain dicts
    content = getattr(last, "content", None) or (
        last.get("content") if isinstance(last, dict) else None
    )
    return content if isinstance(content, str) else None


def _extract_tool_detail(event: dict) -> str:
    """Build a human-readable one-liner for a tool invocation."""
    tool_name = event.get("name", "unknown_tool")
    run_input = event.get("data", {}).get("input", {})
    if not isinstance(run_input, dict):
        run_input = {}

    if tool_name == "semantic_search":
        query = run_input.get("query", "")
        if query:
            formatted_query = _format_ui_query(query)
            return f'• Searching knowledge base for: "{formatted_query}"'
        return "Running semantic search..."
    if tool_name == "page_lookup":
        page = run_input.get("page_number", "?")
        return f"Looking up page {page}..."
    if tool_name == "table_lookup":
        return "Looking up table data..."
    return f"Executing {tool_name}..."


def _format_ui_query(raw_query: str, max_words: int = 5) -> str:
    """Normalize verbose retrieval queries into short UI-facing intent fragments."""
    clean_text = re.sub(r"\s+", " ", raw_query.strip()).strip("\"' ")
    if not clean_text:
        return ""

    words = clean_text.split(" ")
    if len(words) <= max_words:
        return clean_text
    return " ".join(words[:max_words]) + "..."


def _semantic_intent_key(event: dict) -> str:
    """Return a normalized semantic-search intent key for dedupe, else empty string."""
    if event.get("name", "") != "semantic_search":
        return ""
    run_input = event.get("data", {}).get("input", {})
    if not isinstance(run_input, dict):
        return ""
    query = run_input.get("query", "")
    if not isinstance(query, str) or not query.strip():
        return ""
    return _format_ui_query(query)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def adapt_graph_stream(
    event_stream: AsyncIterator[dict],
    response_format: str = "markdown",
) -> AsyncIterator[str]:
    """Consume ``astream_events(v2)`` and yield SSE content strings.

    Protocol
    --------
    1. Yields ``<think>…</think>`` block with live status from graph nodes.
    2. Buffers the final JSON silently.
    3. After the stream exhausts, formats and yields the answer content.

    Parameters
    ----------
    event_stream:
        The async iterator returned by ``CompiledStateGraph.astream_events()``.
    response_format:
        ``"markdown"`` (default) or ``"json"``.
    """
    state = _StreamState()

    try:
        async for event in event_stream:
            event_type = event.get("event", "")
            event_name = event.get("name", "")

            # ── Status events → <think> channel ──────────────────
            if event_type == "on_chain_start":
                status = _NODE_STATUS_MAP.get(event_name)
                if status and status[0].value not in state.phases_emitted:
                    phase, message = status
                    state.phase = phase
                    state.phases_emitted.add(phase.value)

                    if not state.think_open:
                        yield "<think>\n"
                        state.think_open = True
                    yield f"{message}\n"

            # Tool invocations → detailed <think> line
            elif event_type == "on_tool_start":
                if not state.think_open:
                    yield "<think>\n"
                    state.think_open = True
                detail = _extract_tool_detail(event)
                semantic_key = _semantic_intent_key(event)
                if semantic_key:
                    if semantic_key == state.last_intent_line:
                        continue
                    state.last_intent_line = semantic_key
                else:
                    state.last_intent_line = ""
                yield f"{detail}\n"

            # Verifier outcome → <think> feedback
            elif event_type == "on_chain_end" and event_name == NodeName.SELF_CORRECTING_AGENT:
                output = event.get("data", {}).get("output", {})
                if isinstance(output, dict):
                    verifier = output.get("verifier_response")
                    if verifier is not None:
                        grounded = (
                            getattr(verifier, "grounded", None)
                            if not isinstance(verifier, dict)
                            else verifier.get("grounded")
                        )
                        if not state.think_open:
                            yield "<think>\n"
                            state.think_open = True
                        if grounded:
                            yield "Answer verified as grounded ✓\n"
                        else:
                            state.retry_count += 1
                            yield (
                                f"Revising draft to strictly match source documents  "
                                f"(attempt {state.retry_count + 1})...\n"
                            )

            # ── Terminal node output → capture final JSON ─────────
            elif event_type == "on_chain_end" and event_name in _TERMINAL_NODES:
                output = event.get("data", {}).get("output", {})
                if isinstance(output, dict):
                    content = _extract_message_content(output.get("messages", []))
                    if content:
                        try:
                            state.final_json = json.loads(content)
                        except (json.JSONDecodeError, TypeError):
                            state.raw_fallback = content

    except Exception:
        logger.exception("stream_adapter_error")
        # Guarantee the <think> block is closed before the error message
        if state.think_open:
            yield "</think>\n"
            state.think_open = False
        yield "\n\n*[System Error: completion failed.]*"
        return

    # ── Close <think> and emit formatted answer ──────────────────
    if state.think_open:
        yield "</think>\n"

    if state.final_json:
        if response_format == "json":
            yield ChatPresenter.to_json_string(state.final_json)
        else:
            yield ChatPresenter.to_markdown(state.final_json)
    elif state.raw_fallback:
        yield state.raw_fallback
