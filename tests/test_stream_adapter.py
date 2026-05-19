import json

from app.services.stream_adapter import _extract_tool_detail, _format_ui_query, adapt_graph_stream
from rag_core.genetation.nodes import NodeName


def test_format_ui_query_truncates_to_five_words():
    raw = '"Adjusted Free Cash Flow Productivity 105% non-cash charges excluded"'
    assert _format_ui_query(raw) == "Adjusted Free Cash Flow Productivity..."


def test_format_ui_query_keeps_short_text():
    raw = "Beauty segment volume"
    assert _format_ui_query(raw) == "Beauty segment volume"


def test_extract_tool_detail_semantic_search_uses_bullet():
    event = {
        "name": "semantic_search",
        "data": {"input": {"query": "Beauty segment unit volume unchanged foreign exchange"}},
    }
    msg = _extract_tool_detail(event)
    assert msg.startswith('• Searching knowledge base for: "Beauty segment unit volume unchanged..."')


async def _iter_events(events):
    for event in events:
        yield event


def test_adapt_graph_stream_dedupes_identical_consecutive_intents():
    events = [
        {"event": "on_chain_start", "name": NodeName.TOOLS},
        {
            "event": "on_tool_start",
            "name": "semantic_search",
            "data": {"input": {"query": "Beauty segment unit volume unchanged foreign exchange"}},
        },
        {
            "event": "on_tool_start",
            "name": "semantic_search",
            "data": {"input": {"query": "Beauty segment unit volume unchanged foreign exchange"}},
        },
        {
            "event": "on_chain_end",
            "name": NodeName.FALLBACK_HANDLER,
            "data": {
                "output": {
                    "messages": [
                        {
                            "content": json.dumps(
                                {
                                    "answer": "fallback",
                                    "citations": [],
                                }
                            )
                        }
                    ]
                }
            },
        },
    ]

    async def collect():
        chunks = []
        async for chunk in adapt_graph_stream(_iter_events(events), response_format="json"):
            chunks.append(chunk)
        return "".join(chunks)

    import asyncio

    full = asyncio.run(collect())
    assert full.count('• Searching knowledge base for: "Beauty segment unit volume unchanged..."') == 1
    assert full.count("<think>") == 1
    assert full.count("</think>") == 1
