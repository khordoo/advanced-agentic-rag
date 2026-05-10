"""Common embedding model contract."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbeddingModel(ABC):
    """Interface for embedding providers used by indexers and retrievers."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Batch embed multiple strings for indexing."""
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string for retrieval."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the expected vector size."""
        raise NotImplementedError
