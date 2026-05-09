from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document


class BaseRetriever(ABC):
    # todo: not a good name , we will be doing  a hybrid search
    @abstractmethod
    def search(
        self,
        query: str,
        scopes: list[str] | None = None,
        retrival_top_k: int | None = None,
        rerank_top_k: int | None = None,
        rerank: bool = True,
    ) -> List[Document]:
        """Executes a semantic search with hybrid search and reranking."""
        raise NotImplementedError

    @abstractmethod
    def exact_lookup(
        self, filters: dict, scopes: list[str] = None, limit: int = None
    ) -> list[Document]:
        """Executes a deterministic metadata lookup."""
        raise NotImplementedError
