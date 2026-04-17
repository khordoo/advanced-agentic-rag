from abc import ABC, abstractmethod

from langchain_core.documents import Document


class BaseReranker(ABC):
    """
    Base class for rerankers.
    """

    @abstractmethod
    def rerank(
        self, query: str, documents: list[Document], top_k: int | None = None
    ) -> list[Document]:
        """
        Rerank documents based on query relevance.

        Args:
            query: The query string
            documents: List of LangChain Documents to rerank
            top_k: Optional limit on the number of results to return

        Returns:
            Reranked list of Documents (with ``rerank_score`` in metadata).
        """
        raise NotImplementedError
