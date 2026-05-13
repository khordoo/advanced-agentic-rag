from __future__ import annotations

import logging

from rag_core.core.config import RetrievalSettings, get_retrieval_settings
from rag_core.retrieval.base import BaseRetriever
from rag_core.retrieval.postgres_retriever import PostgresRetriever
from rag_core.retrieval.qdrant_retriever import QdrantRetriever

logger = logging.getLogger(__name__)


def create_retriever(settings: RetrievalSettings | None = None) -> BaseRetriever:
    resolved = settings or get_retrieval_settings()
    if resolved.backend == "postgres":
        retriever = PostgresRetriever(settings=resolved)
    else:
        retriever = QdrantRetriever(retrieval_settings=resolved)

    logger.info(
        "Retriever selected (backend=%s, retriever=%s)",
        resolved.backend,
        retriever.__class__.__name__,
    )
    return retriever
