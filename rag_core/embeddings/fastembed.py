"""FastEmbed-backed embedding model implementation."""

from __future__ import annotations

import logging
from typing import Any

from rag_core.embeddings.base import BaseEmbeddingModel

logger = logging.getLogger(__name__)


def _get_fastembed_model(model_name: str) -> Any:
    """Lazy-load a fastembed TextEmbedding model."""
    from fastembed import TextEmbedding  # type: ignore[import-untyped]

    return TextEmbedding(model_name=model_name)


class FastEmbedEmbeddingModel(BaseEmbeddingModel):
    """Embedding adapter for fastembed.TextEmbedding."""

    def __init__(self, model_name: str, dimension: int) -> None:
        if dimension < 1:
            raise ValueError("Embedding dimension must be greater than zero.")
        self.model_name = model_name
        self._dimension = dimension
        self._model = None

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model(self) -> Any:
        if self._model is None:
            logger.info("Loading fastembed model '%s' ...", self.model_name)
            self._model = _get_fastembed_model(self.model_name)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        embeddings = list(self.model.embed(texts))
        vectors = [self._to_float_list(embedding) for embedding in embeddings]
        self._validate_vectors(vectors)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    @staticmethod
    def _to_float_list(embedding: Any) -> list[float]:
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()
        return [float(value) for value in embedding]

    def _validate_vectors(self, vectors: list[list[float]]) -> None:
        for index, vector in enumerate(vectors):
            if len(vector) != self._dimension:
                raise ValueError(
                    "Embedding dimension mismatch at index "
                    f"{index}: model produced {len(vector)} values but "
                    f"configuration expects {self._dimension}."
                )
