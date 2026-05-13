"""End-to-end RAG pipeline: query → retrieve → generate."""

from __future__ import annotations

import logging

from rag_core.agents import AgentFactory
from rag_core.retrival import BaseRetriever, RAGSearchService
from rag_core.prompts import AgentPrompt


logger = logging.getLogger(__name__)

#TODO: rename this to generation
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
        retriever: BaseRetriever | None = None,
        search_service: RAGSearchService | None = None,
        prompt_name: str = "primary_rag_agent",
        # reranker: BaseReranker | None = None,
    ) -> None:
        self.config = config
        self.llm = AgentFactory.get_judge_model(provider, config=config)
        self.search_service = search_service or RAGSearchService(retriever=retriever)
        self.prompt_template = AgentPrompt.from_package(prompt_name).chat_template
        # self.reranker = reranker or LocalReranker()
        logger.info("RAG pipeline initialised (provider=%s)", provider)

    def run(self, query: str) -> dict:
        """Execute the full RAG pipeline for a single query.

        Returns
        -------
        dict
            ``{"response": str, "contexts": list[dict]}``
            Each context dict has ``{"content": str, "metadata": dict}``.
        """
        # context_docs = self.retriever.hierarchical_search(
        #     query,
        #     k=10,
        #     top_n=int(self.config.get('RERANK_TOP_N',3)))
        context_docs = self.search_service.search(query)
        # print("---" * 50)
        # print(context_docs)
        # print("---" * 50)
        for d in context_docs:
            print(d.metadata["score"])
            print("---" * 50)
        context_text = "\n\n----\n\n".join(d.page_content for d in context_docs)
        runtime_vars = {"context": context_text, "question": query}
        prompt_inputs = {
            key: value
            for key, value in runtime_vars.items()
            if key in self.prompt_template.input_variables
        }
        prompt = self.prompt_template.invoke(prompt_inputs)

        response = self.llm.invoke(prompt.to_messages())

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
    # query = "According to the 'Net Sales Change Drivers 2024 vs. 2023' table, what was the specific percentage impact of foreign exchange on the net sales growth for the Grooming segment?"
    # query = "According to the 'Net Sales Change Drivers 2024 vs. 2023' table, which specific reportable segment experienced a 2% favorable 'Mix' impact on its net sales growth?"
    # "ground_truth": "The Health Care segment experienced a 2% favorable Mix impact on its net sales growth."
    # quer = "find something about Net Sales Change Drivers?"

    pipeline = RAGPipeline(cfg, provider="aws")
    result = pipeline.run(query)

    print("\n--- Response ---")
    print(result["response"])
    print("----Sources-----------")
    for r in result["contexts"]:
        print(r.metadata)
    print(f"\n({len(result['contexts'])} context chunks used)")
