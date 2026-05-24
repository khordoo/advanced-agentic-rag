from __future__ import annotations

import pytest

from rag_core.genetation import tools
from rag_core.genetation.tools import AgentToolFactory


def test_create_tools_does_not_initialize_search_service_until_tool_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_on_init(*args, **kwargs):
        raise AssertionError("RAGSearchService should be lazy")

    monkeypatch.setattr(tools, "RAGSearchService", _raise_on_init)

    created_tools = AgentToolFactory.create_tools()

    assert [tool.name for tool in created_tools] == [
        "semantic_search",
        "page_lookup",
        "table_lookup",
    ]

