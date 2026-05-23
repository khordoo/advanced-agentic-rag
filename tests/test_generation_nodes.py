from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from rag_core.genetation.nodes import GraphNodes


class _StubReasoningLLM:
    def bind_tools(self, tools):
        return self


class _StubFastLLM:
    def __init__(self, content):
        self._content = content

    def invoke(self, _messages):
        return SimpleNamespace(content=self._content)


class _StubVerifierLLM:
    def invoke(self, _messages):
        return SimpleNamespace(content='{"grounded": true, "reason": "ok"}')


class _StubSettings:
    general_assistant_lookback = 1
    enable_intent_routing = True
    enable_grounding_verifier = True


def test_general_assistant_normalizes_list_content_to_string():
    list_content = [{"type": "text", "text": "Hi there!", "index": 0}]
    nodes = GraphNodes(
        reasoning_llm=_StubReasoningLLM(),
        verifier_llm=_StubVerifierLLM(),
        fast_llm=_StubFastLLM(list_content),
        tools=[],
        settings=_StubSettings(),
    )

    state = {"messages": [HumanMessage(content="hi")]}  # minimal state for this node
    out = nodes.execute_general_assistant_agent(state)

    payload = out["messages"][0].content
    assert '"answer":"Hi there!"' in payload
