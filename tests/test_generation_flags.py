import json

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool

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
