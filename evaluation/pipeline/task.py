"""Phoenix Experiments task function: runs RAG for a single example.

The ``RagService`` is **not** imported at module level — it is injected
via ``functools.partial`` from the flow body so it is constructed once
per flow run, not once per example.

Top-level imports: stdlib only.
"""

from __future__ import annotations

import asyncio
import logging

from evaluation.pipeline.adapter import get_example_value
from evaluation.pipeline.adapter import strip_file_refs

logger = logging.getLogger(__name__)


def run_rag_for_example(example: dict, *, rag_service) -> dict:
    """Execute the production RAG graph for a single evaluation example.

    Wraps the async inference in asyncio.run to be compatible with Phoenix's
    sync experiment runner.
    """
    return asyncio.run(_run_rag_for_example_async(example, rag_service=rag_service))


async def _run_rag_for_example_async(example: dict, *, rag_service) -> dict:
    """Internal async implementation of the RAG task.

    Parameters
    ----------
    example:
        Phoenix Dataset row.  Must contain ``input`` (the question).
    rag_service:
        A pre-built ``RagService`` instance (injected via partial).

    Returns
    -------
    dict
        ``{response, retrieved_contexts}`` on success, or
        ``{response: "", retrieved_contexts: [], error: str}`` on failure.

    Per-example error isolation: catches ``Exception`` and returns a
    failure dict so Phoenix records the error without aborting the
    entire experiment.
    """
    question = get_example_value(example, "input", "")
    if not question:
        logger.warning("Example has no 'input' key — skipping")
        return {"response": "", "retrieved_contexts": [], "error": "Missing input"}

    try:
        from app.schemas.openai import ChatCompletionRequest, ChatMessage

        chat_req = ChatCompletionRequest(
            model="eval-runner",
            messages=[ChatMessage(role="user", content=question)],
        )

        result = await rag_service.generate_chat_completion(
            chat_req, include_source_docs=True
        )

        # Extract answer from the RagFinalResponseSchema JSON
        response = strip_file_refs(result.agent_output.get("answer", ""))

        # Extract raw page_content strings from source Documents
        retrieved_contexts = []
        for doc in result.source_docs:
            if hasattr(doc, "page_content"):
                retrieved_contexts.append(doc.page_content)
            elif isinstance(doc, str):
                retrieved_contexts.append(doc)
            elif isinstance(doc, dict):
                retrieved_contexts.append(doc.get("content", str(doc)))

        return {
            "response": response,
            "retrieved_contexts": retrieved_contexts,
        }

    except Exception as exc:
        logger.error(
            "Error processing example (question=%s…): %s",
            question[:80],
            exc,
            exc_info=True,
        )
        return {
            "response": "",
            "retrieved_contexts": [],
            "error": str(exc),
        }
