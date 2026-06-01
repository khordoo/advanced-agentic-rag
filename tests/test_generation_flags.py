import json

import pytest

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import ValidationError

from rag_core.core.config import GenerationSettings
from rag_core.genetation.graph import RagGraphFactory
from rag_core.genetation.nodes import GraphNodes, NodeName
from rag_core.prompts.packages.intent_router.schema import IntentCategory
from rag_core.genetation.tools import (
    AgentToolFactory,
    FetchPageToolSchema,
    TableLookupToolSchema,
)
from rag_core.retrieval.search_service import RAGSearchService
from rag_core.schemas import VectorDatabasePayloadKeys as PayloadKeys


def _noop_tool() -> str:
    return "ok"


class _DummyModel:
    def __init__(self, response_content: str = "ok") -> None:
        self.response_content = response_content
        self.invoke_calls = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.invoke_calls += 1
        return AIMessage(content=self.response_content)


def _make_tool():
    return StructuredTool.from_function(
        _noop_tool, name="noop_tool", description="No-op test tool"
    )


class _DummyToolExecutor:
    def __init__(self, tool_messages):
        self.tool_messages = tool_messages

    def invoke(self, payload):
        return {"messages": self.tool_messages}


class _DummyRetriever:
    def __init__(self) -> None:
        self.search_calls = []
        self.lookup_calls = []

    def search(self, query, asset_key=None, **kwargs):
        self.search_calls.append({"query": query, "asset_key": asset_key, **kwargs})
        return [
            Document(
                page_content="search result",
                metadata={PayloadKeys.ASSET_KEY.value: asset_key or "asset-search"},
            )
        ]

    def exact_lookup(self, asset_key=None, filters=None, **kwargs):
        self.lookup_calls.append(
            {"asset_key": asset_key, "filters": filters, **kwargs}
        )
        return [
            Document(
                page_content="lookup result",
                metadata={PayloadKeys.ASSET_KEY.value: asset_key},
            )
        ]


class _NoopReranker:
    def rerank(self, query, documents, top_k=None):
        for i, doc in enumerate(documents, start=1):
            doc.metadata["rerank_score"] = 1.0 / i
        return documents[:top_k] if top_k is not None else documents


def test_initial_router_uses_intent_router_when_enabled():
    model = _DummyModel()
    nodes = GraphNodes(
        reasoning_llm=model,
        verifier_llm=model,
        fast_llm=model,
        tools=[_make_tool()],
        settings=GenerationSettings(enable_intent_routing=True),
    )

    route = nodes.route_initial_node(
        {"messages": [], "source_docs": [], "token_budget": 0, "intent_route": ""}
    )
    assert route == NodeName.INTENT_ROUTER_AGENT


def test_initial_router_skips_intent_router_when_disabled():
    model = _DummyModel()
    nodes = GraphNodes(
        reasoning_llm=model,
        verifier_llm=model,
        fast_llm=model,
        tools=[_make_tool()],
        settings=GenerationSettings(enable_intent_routing=False),
    )

    route = nodes.route_initial_node(
        {"messages": [], "source_docs": [], "token_budget": 0, "intent_route": ""}
    )
    assert route == NodeName.PRIMARY_RAG_AGENT


def test_graph_has_conditional_initial_routes():
    model = _DummyModel()
    app = RagGraphFactory.build(
        reasoning_llm=model,
        verifier_llm=model,
        fast_llm=model,
        tools=[_make_tool()],
        settings=GenerationSettings(enable_intent_routing=True),
    )
    start_edges = [
        edge for edge in app.get_graph().edges if edge.source == "__start__"
    ]
    start_targets = {edge.target for edge in start_edges}

    assert NodeName.INTENT_ROUTER_AGENT in start_targets
    assert NodeName.PRIMARY_RAG_AGENT in start_targets
    assert all(edge.conditional for edge in start_edges)


def test_self_correcting_agent_skips_verifier_when_disabled():
    reasoning_model = _DummyModel()
    verifier_model = _DummyModel(
        response_content=json.dumps({"grounded": False, "reason": "should not run"})
    )
    fast_model = _DummyModel()
    nodes = GraphNodes(
        reasoning_llm=reasoning_model,
        verifier_llm=verifier_model,
        fast_llm=fast_model,
        tools=[_make_tool()],
        settings=GenerationSettings(enable_grounding_verifier=False),
    )

    result = nodes.execute_self_correcting_agent(
        {
            "messages": [HumanMessage(content="q"), AIMessage(content="candidate answer")],
            "source_docs": [Document(page_content="doc", metadata={})],
            "token_budget": 0,
            "intent_route": "domain_expert",
            "self_correction_attempts": 0,
        }
    )

    assert result["verifier_response"].grounded is True
    assert "disabled" in result["verifier_response"].reason.lower()
    assert verifier_model.invoke_calls == 0


def test_self_correcting_agent_invokes_verifier_when_enabled():
    reasoning_model = _DummyModel()
    verifier_model = _DummyModel(
        response_content=json.dumps({"grounded": True, "reason": "looks grounded"})
    )
    fast_model = _DummyModel()
    nodes = GraphNodes(
        reasoning_llm=reasoning_model,
        verifier_llm=verifier_model,
        fast_llm=fast_model,
        tools=[_make_tool()],
        settings=GenerationSettings(enable_grounding_verifier=True),
    )

    result = nodes.execute_self_correcting_agent(
        {
            "messages": [HumanMessage(content="q"), AIMessage(content="candidate answer")],
            "source_docs": [Document(page_content="doc", metadata={})],
            "token_budget": 0,
            "intent_route": "domain_expert",
            "self_correction_attempts": 0,
        }
    )

    assert result["verifier_response"].grounded is True
    assert result["verifier_response"].reason == "looks grounded"
    assert verifier_model.invoke_calls == 1


def test_self_correcting_agent_marks_failed_verification_span(monkeypatch):
    reasoning_model = _DummyModel()
    verifier_model = _DummyModel(
        response_content=json.dumps({"grounded": False, "reason": "unsupported claim"})
    )
    fast_model = _DummyModel()
    nodes = GraphNodes(
        reasoning_llm=reasoning_model,
        verifier_llm=verifier_model,
        fast_llm=fast_model,
        tools=[_make_tool()],
        settings=GenerationSettings(enable_grounding_verifier=True),
    )

    class _RecordingSpan:
        def __init__(self) -> None:
            self.attributes = {}

        def set_attribute(self, key, value):
            self.attributes[key] = value

    class _RecordingTracer:
        def __init__(self) -> None:
            self.span = _RecordingSpan()
            self.span_name = None

        def start_as_current_span(self, name, context=None):
            self.span_name = name
            return self

        def __enter__(self):
            return self.span

        def __exit__(self, exc_type, exc, tb):
            return False

    tracer = _RecordingTracer()
    span = _RecordingSpan()
    monkeypatch.setattr(
        "rag_core.core.telemetry.trace.get_current_span",
        lambda: span,
    )
    monkeypatch.setattr("rag_core.core.telemetry.tracer", tracer)

    result = nodes.execute_self_correcting_agent(
        {
            "messages": [HumanMessage(content="q"), AIMessage(content="candidate answer")],
            "source_docs": [Document(page_content="doc", metadata={})],
            "token_budget": 0,
            "intent_route": "domain_expert",
            "self_correction_attempts": 0,
        }
    )

    assert result["verifier_response"].grounded is False
    assert span.attributes == {}
    assert tracer.span_name == "agent.verification.rejected"
    assert tracer.span.attributes["custom.verification_status"] == "failed"
    assert tracer.span.attributes["custom.verification_grounded"] is False
    assert "unsupported" in tracer.span.attributes["custom.verification_reason"]
    # Eval-ready fields
    assert tracer.span.attributes["question"] == "q"
    assert tracer.span.attributes["answer"] == "candidate answer"
    assert tracer.span.attributes["contexts"] == ["doc"]


def test_self_correcting_agent_skips_feedback_when_extracting_question(monkeypatch):
    """When verifier feedback messages precede the answer, the marker span
    should capture the *original* user question, not the feedback."""
    reasoning_model = _DummyModel()
    verifier_model = _DummyModel(
        response_content=json.dumps({"grounded": False, "reason": "still wrong"})
    )
    fast_model = _DummyModel()
    nodes = GraphNodes(
        reasoning_llm=reasoning_model,
        verifier_llm=verifier_model,
        fast_llm=fast_model,
        tools=[_make_tool()],
        settings=GenerationSettings(enable_grounding_verifier=True),
    )

    class _RecordingSpan:
        def __init__(self) -> None:
            self.attributes = {}

        def set_attribute(self, key, value):
            self.attributes[key] = value

    class _RecordingTracer:
        def __init__(self) -> None:
            self.span = _RecordingSpan()
            self.span_name = None

        def start_as_current_span(self, name, context=None):
            self.span_name = name
            return self

        def __enter__(self):
            return self.span

        def __exit__(self, exc_type, exc, tb):
            return False

    tracer = _RecordingTracer()
    monkeypatch.setattr("rag_core.core.telemetry.tracer", tracer)

    result = nodes.execute_self_correcting_agent(
        {
            "messages": [
                HumanMessage(content="What was the revenue?"),
                AIMessage(content="first attempt"),
                HumanMessage(
                    content=f"{GraphNodes.VERIFIER_REJECTION_PREFIX} Reason: bad."
                ),
                AIMessage(content="second attempt"),
            ],
            "source_docs": [Document(page_content="revenue data", metadata={})],
            "token_budget": 0,
            "intent_route": "domain_expert",
            "self_correction_attempts": 1,
        }
    )

    assert result["verifier_response"].grounded is False
    # Must skip the feedback HumanMessage and find the original question
    assert tracer.span.attributes["question"] == "What was the revenue?"
    assert tracer.span.attributes["answer"] == "second attempt"
    assert tracer.span.attributes["contexts"] == ["revenue data"]


def test_generation_settings_enforce_safe_minimums():
    with pytest.raises(ValidationError):
        GenerationSettings(max_agent_iterations=4)

    with pytest.raises(ValidationError):
        GenerationSettings(max_tool_calls=2)

    with pytest.raises(ValidationError):
        GenerationSettings(max_self_correction_attempts=0)


def test_generation_tool_schemas_require_asset_key():
    with pytest.raises(ValidationError):
        FetchPageToolSchema(file_name="Annual Report.pdf", page_number=3)

    with pytest.raises(ValidationError):
        TableLookupToolSchema(table_id="DATA_TABLE_001", file_name="Annual Report.pdf")

    assert FetchPageToolSchema(asset_key="asset-42", page_number=3).asset_key == "asset-42"
    assert (
        TableLookupToolSchema(table_id="DATA_TABLE_001", asset_key="asset-42").asset_key
        == "asset-42"
    )


def test_generation_lookup_tools_filter_by_asset_key():
    retriever = _DummyRetriever()
    factory = AgentToolFactory(
        search_service=RAGSearchService(retriever=retriever, reranker=_NoopReranker())
    )

    factory._page_lookup(asset_key="asset-42", page_number=3)
    factory._table_lookup(table_id="DATA_TABLE_001", asset_key="asset-42")

    assert retriever.lookup_calls == [
        {
            "asset_key": "asset-42",
            "filters": {PayloadKeys.PAGE.value: [3]},
        },
        {
            "asset_key": "asset-42",
            "filters": {PayloadKeys.TABLE_ID.value: "DATA_TABLE_001"},
        },
    ]


def test_semantic_search_tool_passes_optional_asset_key():
    retriever = _DummyRetriever()
    factory = AgentToolFactory(
        search_service=RAGSearchService(retriever=retriever, reranker=_NoopReranker())
    )

    factory._semantic_search(query="revenue", asset_key="asset-42")

    assert retriever.search_calls == [
        {
            "query": "revenue",
            "asset_key": "asset-42",
            "scopes": None,
            "filters": None,
            "retrieval_top_k": None,
            "hybrid_alpha": None,
            "rerank": False,
        }
    ]


def test_intent_router_handles_bedrock_list_content():
    fast_model = _DummyModel(
        response_content=[
            {
                "text": json.dumps(
                    {"reasoning": "needs retrieval", "intent": IntentCategory.DOMAIN_EXPERT.value}
                )
            }
        ]
    )
    nodes = GraphNodes(
        reasoning_llm=_DummyModel(),
        verifier_llm=_DummyModel(),
        fast_llm=fast_model,
        tools=[_make_tool()],
        settings=GenerationSettings(enable_intent_routing=True),
    )

    result = nodes.execute_intent_router_agent(
        {
            "messages": [HumanMessage(content="What was revenue last quarter?")],
            "source_docs": [],
            "token_budget": 0,
            "intent_route": "",
        }
    )

    assert result["intent_route"] == IntentCategory.DOMAIN_EXPERT.value


def test_intent_router_falls_back_on_malformed_content():
    fast_model = _DummyModel(response_content=[{"text": "not-json"}])
    nodes = GraphNodes(
        reasoning_llm=_DummyModel(),
        verifier_llm=_DummyModel(),
        fast_llm=fast_model,
        tools=[_make_tool()],
        settings=GenerationSettings(enable_intent_routing=True),
    )

    result = nodes.execute_intent_router_agent(
        {
            "messages": [HumanMessage(content="Hello?")],
            "source_docs": [],
            "token_budget": 0,
            "intent_route": "",
        }
    )

    assert result["intent_route"] == IntentCategory.GENERAL_ASSISTANT.value


def test_primary_rag_agent_increments_global_iteration_count():
    model = _DummyModel(response_content="answer")
    nodes = GraphNodes(
        reasoning_llm=model,
        verifier_llm=model,
        fast_llm=model,
        tools=[_make_tool()],
        settings=GenerationSettings(),
    )

    result = nodes.execute_primary_rag_agent(
        {
            "messages": [HumanMessage(content="q")],
            "system_instructions": "Be helpful.",
            "source_docs": [],
            "token_budget": 0,
            "intent_route": "domain_expert",
            "iteration_count": 2,
        }
    )
    assert result["iteration_count"] == 3


def test_tool_router_uses_global_kill_switch_before_tool_routing():
    model = _DummyModel()
    nodes = GraphNodes(
        reasoning_llm=model,
        verifier_llm=model,
        fast_llm=model,
        tools=[_make_tool()],
        settings=GenerationSettings(max_agent_iterations=5),
    )

    route = nodes.tool_router(
        {
            "messages": [AIMessage(content="final answer")],
            "source_docs": [],
            "token_budget": 0,
            "intent_route": "domain_expert",
            "iteration_count": 5,
        }
    )
    assert route == NodeName.FALLBACK_HANDLER


def test_tool_router_routes_to_finish_without_tool_calls():
    model = _DummyModel()
    nodes = GraphNodes(
        reasoning_llm=model,
        verifier_llm=model,
        fast_llm=model,
        tools=[_make_tool()],
        settings=GenerationSettings(max_agent_iterations=25),
    )

    route = nodes.tool_router(
        {
            "messages": [AIMessage(content="final answer")],
            "source_docs": [],
            "token_budget": 0,
            "intent_route": "domain_expert",
            "iteration_count": 1,
            "tool_call_count": 0,
        }
    )
    assert route == NodeName.FINISH


def test_tool_router_enforces_tool_call_budget():
    model = _DummyModel()
    nodes = GraphNodes(
        reasoning_llm=model,
        verifier_llm=model,
        fast_llm=model,
        tools=[_make_tool()],
        settings=GenerationSettings(max_agent_iterations=25, max_tool_calls=3),
    )
    msg = AIMessage(
        content="need tools",
        tool_calls=[{"name": "noop_tool", "args": {}, "id": "call-1", "type": "tool_call"}],
    )

    route = nodes.tool_router(
        {
            "messages": [msg],
            "source_docs": [],
            "token_budget": 0,
            "intent_route": "domain_expert",
            "iteration_count": 1,
            "tool_call_count": 3,
        }
    )
    assert route == NodeName.FALLBACK_HANDLER


def test_execute_tools_increments_tool_call_count():
    model = _DummyModel()
    nodes = GraphNodes(
        reasoning_llm=model,
        verifier_llm=model,
        fast_llm=model,
        tools=[_make_tool()],
        settings=GenerationSettings(),
    )
    nodes.tool_executor = _DummyToolExecutor(
        [
            ToolMessage(content="[]", tool_call_id="call-1"),
            ToolMessage(content="[]", tool_call_id="call-2"),
        ]
    )
    msg = AIMessage(
        content="need tools",
        tool_calls=[
            {"name": "noop_tool", "args": {}, "id": "call-1", "type": "tool_call"},
            {"name": "noop_tool", "args": {}, "id": "call-2", "type": "tool_call"},
        ],
    )

    result = nodes.execute_tools(
        {
            "messages": [msg],
            "source_docs": [],
            "token_budget": 0,
            "intent_route": "domain_expert",
            "tool_call_count": 4,
        }
    )
    assert result["tool_call_count"] == 6


def test_self_correcting_agent_handles_list_answer_content():
    reasoning_model = _DummyModel()
    verifier_model = _DummyModel(
        response_content=json.dumps({"grounded": True, "reason": "looks grounded"})
    )
    fast_model = _DummyModel()
    nodes = GraphNodes(
        reasoning_llm=reasoning_model,
        verifier_llm=verifier_model,
        fast_llm=fast_model,
        tools=[_make_tool()],
        settings=GenerationSettings(enable_grounding_verifier=True),
    )

    result = nodes.execute_self_correcting_agent(
        {
            "messages": [
                HumanMessage(content="q"),
                AIMessage(
                    content=[
                        {"type": "text", "text": "I cannot find the answer."},
                        {
                            "type": "reasoning_content",
                            "reasoning_content": {"text": "hidden chain-of-thought"},
                        },
                    ]
                ),
            ],
            "source_docs": [Document(page_content="doc", metadata={})],
            "token_budget": 0,
            "intent_route": "domain_expert",
            "self_correction_attempts": 0,
        }
    )

    assert result["verifier_response"].grounded is True
    assert result["verifier_response"].reason == "looks grounded"
    assert verifier_model.invoke_calls == 1


def test_post_processing_handles_list_answer_content():
    model = _DummyModel()
    nodes = GraphNodes(
        reasoning_llm=model,
        verifier_llm=model,
        fast_llm=model,
        tools=[_make_tool()],
        settings=GenerationSettings(),
    )

    result = nodes.execute_post_processing(
        {
            "messages": [
                AIMessage(
                    content=[
                        {
                            "type": "text",
                            "text": "I cannot find the answer in the provided documents.",
                        },
                        {
                            "type": "reasoning_content",
                            "reasoning_content": {"text": "hidden chain-of-thought"},
                        },
                    ]
                )
            ],
            "source_docs": [],
            "token_budget": 0,
            "intent_route": "domain_expert",
        }
    )

    payload = json.loads(result["messages"][0].content)
    assert payload["answer"] == "I cannot find the answer in the provided documents."
    assert payload["citations"] == []


def test_post_processing_matches_citations_by_exact_asset_key():
    model = _DummyModel()
    nodes = GraphNodes(
        reasoning_llm=model,
        verifier_llm=model,
        fast_llm=model,
        tools=[_make_tool()],
        settings=GenerationSettings(),
    )

    result = nodes.execute_post_processing(
        {
            "messages": [
                AIMessage(
                    content=(
                        "Revenue increased [1]. Display-name citations do not match [2]."
                        "\n\n### Sources\n"
                        "[1] asset-42 (Page 3)\n"
                        "[2] Annual Report.pdf (Page 4)"
                    )
                )
            ],
            "source_docs": [
                Document(
                    page_content="doc",
                    metadata={
                        "asset_key": "asset-42",
                        "asset_name": "Annual Report.pdf",
                        "source": "/docs/Annual Report.pdf",
                    },
                )
            ],
            "token_budget": 0,
            "intent_route": "domain_expert",
        }
    )

    payload = json.loads(result["messages"][0].content)
    assert payload["citations"][0]["asset_key"] == "asset-42"
    assert payload["citations"][0]["asset_name"] == "Annual Report.pdf"
    assert payload["citations"][0]["source_url"] == "/docs/Annual Report.pdf"
    assert "file_name" not in payload["citations"][0]
    assert payload["citations"][1]["asset_key"] == "Annual Report.pdf"
    assert payload["citations"][1]["asset_name"] == ""
    assert payload["citations"][1]["source_url"] == ""


def test_post_processing_normalizes_json_answer_with_embedded_sources_footer():
    model = _DummyModel()
    nodes = GraphNodes(
        reasoning_llm=model,
        verifier_llm=model,
        fast_llm=model,
        tools=[_make_tool()],
        settings=GenerationSettings(),
    )

    result = nodes.execute_post_processing(
        {
            "messages": [
                AIMessage(
                    content=json.dumps(
                        {
                            "answer": (
                                "- **Basis-point change:** **200 bps** [1]\n\n"
                                "**Sources**\n"
                                "[1] asset-42 (Page 4)"
                            ),
                            "citations": [],
                        }
                    )
                )
            ],
            "source_docs": [
                Document(
                    page_content="doc",
                    metadata={
                        "asset_key": "asset-42",
                        "asset_name": "Annual Report.pdf",
                        "source": "/docs/Annual Report.pdf",
                    },
                )
            ],
            "token_budget": 0,
            "intent_route": "domain_expert",
        }
    )

    payload = json.loads(result["messages"][0].content)
    assert payload["answer"] == "- **Basis-point change:** **200 bps** [1]"
    assert len(payload["citations"]) == 1
    assert payload["citations"][0]["document_index"] == 1
    assert payload["citations"][0]["asset_key"] == "asset-42"
    assert payload["citations"][0]["asset_name"] == "Annual Report.pdf"
    assert payload["citations"][0]["source_url"] == "/docs/Annual Report.pdf"
    assert payload["citations"][0]["page_number"] == 4


def test_post_processing_dedupes_same_asset_same_page_and_remaps_inline_refs():
    model = _DummyModel()
    nodes = GraphNodes(
        reasoning_llm=model,
        verifier_llm=model,
        fast_llm=model,
        tools=[_make_tool()],
        settings=GenerationSettings(),
    )

    result = nodes.execute_post_processing(
        {
            "messages": [
                AIMessage(
                    content=(
                        "Revenue rose [1][2], while margin changed [12]."
                        "\n\n### Sources\n"
                        "[1] asset-42 (Page 3)\n"
                        "[2] asset-42 (Page 3)\n"
                        "[12] asset-42 (Page 9)"
                    )
                )
            ],
            "source_docs": [
                Document(
                    page_content="doc",
                    metadata={
                        "asset_key": "asset-42",
                        "asset_name": "Annual Report.pdf",
                        "source": "/docs/Annual Report.pdf",
                    },
                )
            ],
            "token_budget": 0,
            "intent_route": "domain_expert",
        }
    )

    payload = json.loads(result["messages"][0].content)
    assert payload["answer"] == "Revenue rose [1], while margin changed [2]."
    assert len(payload["citations"]) == 2
    assert payload["citations"][0]["document_index"] == 1
    assert payload["citations"][0]["asset_key"] == "asset-42"
    assert payload["citations"][0]["page_number"] == 3
    assert payload["citations"][1]["document_index"] == 2
    assert payload["citations"][1]["page_number"] == 9


def test_post_processing_collapses_comma_separated_duplicate_inline_refs():
    model = _DummyModel()
    nodes = GraphNodes(
        reasoning_llm=model,
        verifier_llm=model,
        fast_llm=model,
        tools=[_make_tool()],
        settings=GenerationSettings(),
    )

    result = nodes.execute_post_processing(
        {
            "messages": [
                AIMessage(
                    content=(
                        "Revenue rose [1], [2]."
                        "\n\n### Sources\n"
                        "[1] asset-42 (Page 3)\n"
                        "[2] asset-42 (Page 3)"
                    )
                )
            ],
            "source_docs": [
                Document(
                    page_content="doc",
                    metadata={
                        "asset_key": "asset-42",
                        "asset_name": "Annual Report.pdf",
                        "source": "/docs/Annual Report.pdf",
                    },
                )
            ],
            "token_budget": 0,
            "intent_route": "domain_expert",
        }
    )

    payload = json.loads(result["messages"][0].content)
    assert payload["answer"] == "Revenue rose [1]."
    assert len(payload["citations"]) == 1
    assert payload["citations"][0]["document_index"] == 1


def test_post_processing_keeps_distinct_pages_for_same_asset():
    model = _DummyModel()
    nodes = GraphNodes(
        reasoning_llm=model,
        verifier_llm=model,
        fast_llm=model,
        tools=[_make_tool()],
        settings=GenerationSettings(),
    )

    result = nodes.execute_post_processing(
        {
            "messages": [
                AIMessage(
                    content=(
                        "Two facts [1] and [2]."
                        "\n\n### Sources\n"
                        "[1] asset-42 (Page 3)\n"
                        "[2] asset-42 (Page 4)"
                    )
                )
            ],
            "source_docs": [
                Document(
                    page_content="doc",
                    metadata={
                        "asset_key": "asset-42",
                        "asset_name": "Annual Report.pdf",
                        "source": "/docs/Annual Report.pdf",
                    },
                )
            ],
            "token_budget": 0,
            "intent_route": "domain_expert",
        }
    )

    payload = json.loads(result["messages"][0].content)
    assert payload["answer"] == "Two facts [1] and [2]."
    assert len(payload["citations"]) == 2
    assert payload["citations"][0]["page_number"] == 3
    assert payload["citations"][1]["page_number"] == 4
