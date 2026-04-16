from abc import ABC, abstractmethod


class BaseReranker(ABC):
    """
    Base class for rerankers.
    """

    @abstractmethod
    def rerank(
        self, query: str, documents: list[str], top_k: int
    ) -> list[tuple[str, float]]:
        """
        Rerank documents based on query relevance.

        Args:
            query: The query string
            documents: List of documents to rerank

        Returns:
            List of tuples containing (document, score)
        """
        raise NotImplementedError
