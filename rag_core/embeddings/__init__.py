"""Embedding model abstractions and factories."""

from rag_core.embeddings.base import BaseEmbeddingModel
from rag_core.embeddings.types import EmbeddingProvider

__all__ = [
    "BaseEmbeddingModel",
    "EmbeddingProvider",
    "FastEmbedEmbeddingModel",
    "OpenAIEmbeddingModel",
    "create_embedding_model",
]


def __getattr__(name: str):
    if name == "FastEmbedEmbeddingModel":
        from rag_core.embeddings.fastembed import FastEmbedEmbeddingModel

        return FastEmbedEmbeddingModel
    if name == "OpenAIEmbeddingModel":
        from rag_core.embeddings.openai import OpenAIEmbeddingModel

        return OpenAIEmbeddingModel
    if name == "create_embedding_model":
        from rag_core.embeddings.factory import create_embedding_model

        return create_embedding_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
