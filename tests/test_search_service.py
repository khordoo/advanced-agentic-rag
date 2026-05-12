from langchain_core.documents import Document

from rag_core.core.config import RetrievalSettings, RerankerSettings
from rag_core.retrival.search_service import RAGSearchService


class _FakeRetriever:
    def __init__(self) -> None:
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append({"query": query, **kwargs})
        return [
            Document(page_content="doc-a", metadata={"score": 0.6}),
            Document(page_content="doc-b", metadata={"score": 0.4}),
        ]


class _FakeReranker:
    def rerank(self, query, documents, top_k=None):
        scored = list(reversed(documents))
        for i, doc in enumerate(scored, start=1):
            doc.metadata["rerank_score"] = 1.0 - (i * 0.1)
        return scored[:top_k] if top_k is not None else scored


def test_search_service_stamps_retrieval_provenance_without_rerank() -> None:
    retriever = _FakeRetriever()
    service = RAGSearchService(
        retriever=retriever,
        retrieval_settings=RetrievalSettings(backend="postgres", _env_file=None),
        reranker=_FakeReranker(),
        reranker_settings=RerankerSettings(top_k=3, _env_file=None),
    )

    docs = service.search("q", rerank=False)

    assert docs[0].metadata["retrieval_rank"] == 1
    assert docs[0].metadata["retrieval_score"] == 0.6
    assert docs[0].metadata["retrieval_backend"] == "postgres"
    assert "rerank_rank" not in docs[0].metadata


def test_search_service_preserves_retrieval_score_after_rerank() -> None:
    retriever = _FakeRetriever()
    service = RAGSearchService(
        retriever=retriever,
        retrieval_settings=RetrievalSettings(backend="qdrant", _env_file=None),
        reranker=_FakeReranker(),
        reranker_settings=RerankerSettings(top_k=2, _env_file=None),
    )

    docs = service.search("q", rerank=True)

    assert docs[0].metadata["retrieval_score"] == 0.4
    assert docs[0].metadata["rerank_rank"] == 1
    assert docs[0].metadata["score"] == docs[0].metadata["rerank_score"]
