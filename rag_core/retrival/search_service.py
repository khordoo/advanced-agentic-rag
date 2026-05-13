from __future__ import annotations

from langchain_core.documents import Document

from rag_core.core.config import (
    RetrievalSettings,
    RerankerSettings,
    get_retrieval_settings,
)
from rag_core.reranking import BaseReranker, LocalReranker
from rag_core.retrival.base import BaseRetriever
from rag_core.retrival.factory import create_retriever


class RAGSearchService:
    """Orchestrates retrieval and optional reranking with score provenance."""

    def __init__(
        self,
        retriever: BaseRetriever | None = None,
        retrieval_settings: RetrievalSettings | None = None,
        reranker: BaseReranker | None = None,
        reranker_settings: RerankerSettings | None = None,
    ) -> None:
        self.retrieval_settings = retrieval_settings or get_retrieval_settings()
        self.retriever = retriever or create_retriever(settings=self.retrieval_settings)
        self.reranker = reranker
        self._reranker_settings = reranker_settings or RerankerSettings()

    def search(
        self,
        query: str,
        asset_key: str | None = None,
        scopes: list[str] | None = None,
        filters: dict | None = None,
        retrival_top_k: int | None = None,
        hybrid_alpha: float | None = None,
        rerank_top_k: int | None = None,
        rerank: bool | None = None,
    ) -> list[Document]:
        documents = self.retriever.search(
            query=query,
            asset_key=asset_key,
            scopes=scopes,
            filters=filters,
            retrival_top_k=retrival_top_k,
            hybrid_alpha=hybrid_alpha,
            rerank=False,
        )

        backend_name = self.retrieval_settings.backend
        for i, doc in enumerate(documents, start=1):
            doc.metadata["retrieval_rank"] = i
            doc.metadata["retrieval_score"] = doc.metadata.get("score")
            doc.metadata["retrieval_backend"] = backend_name

        should_rerank = self._reranker_settings.enabled if rerank is None else rerank
        if not should_rerank or not documents:
            return documents

        active_reranker = self.reranker or LocalReranker(settings=self._reranker_settings)
        top_k = rerank_top_k if rerank_top_k is not None else self._reranker_settings.top_k
        reranked = active_reranker.rerank(query, documents, top_k=top_k)

        for i, doc in enumerate(reranked, start=1):
            doc.metadata["rerank_rank"] = i
            rerank_score = doc.metadata.get("rerank_score")
            doc.metadata["score"] = rerank_score

        return reranked
