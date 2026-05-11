"""Placeholder for a future OpenAI embedding provider."""

from __future__ import annotations

from rag_core.embeddings.base import BaseEmbeddingModel


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    """Placeholder adapter for OpenAI embeddings."""

    def __init__(self, model_name: str, dimension: int) -> None:
        raise NotImplementedError(
            "OpenAI embeddings are not implemented yet. "
            "Use EMBEDDING_PROVIDER=fastembed or implement OpenAIEmbeddingModel."
        )

    @property
    def dimension(self) -> int:
        raise NotImplementedError("OpenAI embeddings are not implemented yet.")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        del texts
        raise NotImplementedError("OpenAI embeddings are not implemented yet.")

    def embed_query(self, text: str) -> list[float]:
        del text
        raise NotImplementedError("OpenAI embeddings are not implemented yet.")
