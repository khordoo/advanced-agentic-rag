"""Embedding model abstractions and factories."""

from rag_core.embeddings.base import BaseEmbeddingModel
from rag_core.embeddings.factory import create_embedding_model
from rag_core.embeddings.fastembed import FastEmbedEmbeddingModel
from rag_core.embeddings.openai import OpenAIEmbeddingModel

__all__ = [
    "BaseEmbeddingModel",
    "FastEmbedEmbeddingModel",
    "OpenAIEmbeddingModel",
    "create_embedding_model",
]
