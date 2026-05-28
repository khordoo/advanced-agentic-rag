"""Factory for configured reranker providers."""

from __future__ import annotations

import logging

from rag_core.core.config import RerankerSettings, get_reranker_settings
from rag_core.reranking.base import BaseReranker
from rag_core.reranking.fastembed_reranker import FastEmbedReranker

logger = logging.getLogger(__name__)


def get_reranker(settings: RerankerSettings | None = None) -> BaseReranker:
    """Build the configured reranker provider."""
    active_settings = settings or get_reranker_settings()
    provider = active_settings.provider.strip().lower()

    if provider == "fastembed":
        reranker = FastEmbedReranker(settings=active_settings)
        logger.info(
            "Reranker selected (provider=%s, reranker=%s)",
            provider,
            reranker.__class__.__name__,
        )
        return reranker

    raise ValueError(f"Unsupported reranker provider: {active_settings.provider}")
