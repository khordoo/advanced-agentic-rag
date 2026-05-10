"""Placeholder for a future OpenAI embedding provider."""

from __future__ import annotations

from typing import NoReturn

from rag_core.embeddings.base import BaseEmbeddingModel


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    """Placeholder adapter for OpenAI embeddings."""

    def __init__(self, model_name: str, dimension: int) -> None:
        self.model_name = model_name
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        del texts
        self._raise_not_implemented()

    def embed_query(self, text: str) -> list[float]:
        del text
        self._raise_not_implemented()

    def _raise_not_implemented(self) -> NoReturn:
        raise NotImplementedError(
            "OpenAI embeddings are not implemented yet. "
            "Use EMBEDDING_PROVIDER=fastembed or implement OpenAIEmbeddingModel."
        )
