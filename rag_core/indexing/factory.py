"""Factory for runtime indexer selection based on configuration.

Uses lazy imports so that Qdrant or pgvector dependencies are only loaded
when actually selected, keeping cold-start times low.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_core.indexing.base import BaseIndexer

logger = logging.getLogger(__name__)


class IndexerBackend(str, Enum):
    """Supported vector store backends."""

    QDRANT = "qdrant"
    PGVECTOR = "pgvector"


def create_indexer(backend: IndexerBackend | str | None = None) -> BaseIndexer:
    """Construct the appropriate indexer based on configuration.

    Parameters
    ----------
    backend:
        Explicit backend override.  When *None*, reads from
        ``IngestionSettings.indexer_backend``.

    Returns
    -------
    BaseIndexer
        A fully initialised indexer ready for ``index_asset()`` calls.

    Raises
    ------
    ValueError
        If the backend string is not recognized.
    """
    if backend is None:
        from rag_core.core.config import get_ingestion_settings

        backend = get_ingestion_settings().indexer_backend

    backend = IndexerBackend(backend)

    if backend == IndexerBackend.QDRANT:
        from rag_core.indexing.qdrant_indexer import QdrantIndexer

        logger.info("Indexer backend resolved: Qdrant")
        return QdrantIndexer()

    if backend == IndexerBackend.PGVECTOR:
        from rag_core.indexing.pgvector_indexer import PgVectorIndexer

        logger.info("Indexer backend resolved: pgvector")
        return PgVectorIndexer()

    # Unreachable when using the enum, but guards against future additions.
    raise ValueError(f"Unknown indexer backend: '{backend}'")
