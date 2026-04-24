import json

import pytest

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import ValidationError

from rag_core.core.config import GenerationSettings
from rag_core.genetation.graph import RagGraphFactory
from rag_core.genetation.nodes import GraphNodes, NodeName


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


def test_generation_settings_enforce_safe_minimums():
    with pytest.raises(ValidationError):
        GenerationSettings(max_agent_iterations=4)

    with pytest.raises(ValidationError):
        GenerationSettings(max_tool_calls=2)

    with pytest.raises(ValidationError):
        GenerationSettings(max_self_correction_attempts=0)


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
