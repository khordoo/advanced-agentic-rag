"""End-to-end RAG pipeline: query → retrieve → generate."""

from __future__ import annotations

import logging

from rag_core.agents import AgentFactory, PromptLoader
from rag_core.retriever import Retriever

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Orchestrates retrieval and generation.

    Parameters
    ----------
    config:
        Configuration dict (from :func:`rag_core.config.load_config`).
    provider:
        LLM provider name (``aws``, ``ollama``, ``lmstudio``, ``kimi``).
    retriever:
        Optional pre-built :class:`Retriever`. If *None*, one is created
        from Chroma using the supplied *config*.
    """

    def __init__(
        self,
        config: dict,
        provider: str = "aws",
        retriever: Retriever | None = None,
    ) -> None:
        self.config = config
        self.llm = AgentFactory.create(provider, config=config)
        self.retriever = retriever or Retriever.from_chroma(config)
        logger.info("RAG pipeline initialised (provider=%s)", provider)

    def run(self, query: str) -> dict:
        """Execute the full RAG pipeline for a single query.

        Returns
        -------
        dict
            ``{"response": str, "contexts": list[str]}``
        """
        context_docs = self.retriever.hierarchical_search(query, k=10)
        print('context text:', context_docs)
        context_text = "\n\n----\n\n".join(context_docs)

        prompt = PromptLoader.load("rag_qa", context=context_text, question=query)
        response = self.llm.invoke(prompt.template)

        return {
            "response": response.content,
            "contexts": context_docs,
        }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from rag_core.config import load_config

    logging.basicConfig(level=logging.INFO)
    cfg = load_config()

    query = "What are the key differentiators mentioned in the company's integrated strategy?"
    pipeline = RAGPipeline(cfg, provider="aws")
    result = pipeline.run(query)

    print("\n--- Response ---")
    print(result["response"])
    print(f"\n({len(result['contexts'])} context chunks used)")
