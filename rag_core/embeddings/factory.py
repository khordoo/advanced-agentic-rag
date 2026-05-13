"""Factory for configured embedding model providers."""

from __future__ import annotations

from rag_core.core.config import EmbeddingSettings, get_embedding_settings
from rag_core.embeddings.base import BaseEmbeddingModel
from rag_core.embeddings.fastembed import FastEmbedEmbeddingModel
from rag_core.embeddings.openai import OpenAIEmbeddingModel
from rag_core.embeddings.types import EmbeddingProvider


def create_embedding_model(
    settings: EmbeddingSettings | None = None,
) -> BaseEmbeddingModel:
    """Build the configured embedding model provider."""
    active_settings = settings or get_embedding_settings()
    provider = EmbeddingProvider(active_settings.provider)

    if provider == EmbeddingProvider.FASTEMBED:
        return FastEmbedEmbeddingModel(
            model_name=active_settings.model_name,
            dimension=active_settings.dimension,
        )

    if provider == EmbeddingProvider.OPENAI:
        return OpenAIEmbeddingModel(
            model_name=active_settings.model_name,
            dimension=active_settings.dimension,
        )

    raise ValueError(f"Unsupported embedding provider: {provider}")
