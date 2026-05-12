"""Factory for runtime indexer selection based on configuration.

Uses lazy imports so that Qdrant or Postgres dependencies are only loaded
when actually selected, keeping cold-start times low.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from enum import Enum
from typing import TYPE_CHECKING

from sqlalchemy import text

from rag_core.embeddings.types import EmbeddingProvider

if TYPE_CHECKING:
    from rag_core.indexing.base import BaseIndexer
    from rag_core.store.postgres.session import Database

logger = logging.getLogger(__name__)


class IndexerBackend(str, Enum):
    """Supported vector store backends."""

    QDRANT = "qdrant"
    POSTGRES = "postgres"


@dataclass(frozen=True)
class IndexingCompatibilityResult:
    """Runtime indexing compatibility check result."""

    backend: IndexerBackend
    configured_dimension: int | None = None
    existing_dimensions: tuple[int, ...] = ()
    overwrite_rebuild: bool = False


def _validate_backend_embedding_compatibility(
    backend: IndexerBackend | str | None = None,
) -> IndexerBackend:
    """Validate cross-subsystem embedding compatibility for an indexer backend."""
    if backend is None:
        from rag_core.core.config import get_ingestion_settings

        backend = get_ingestion_settings().indexer_backend

    resolved_backend = IndexerBackend(backend)

    from rag_core.core.config import get_embedding_settings

    provider = EmbeddingProvider(get_embedding_settings().provider)
    if (
        resolved_backend == IndexerBackend.QDRANT
        and provider != EmbeddingProvider.FASTEMBED
    ):
        raise ValueError(
            "Qdrant currently requires QdrantClient FastEmbed dense+sparse models "
            "configured via QDRANT_EMBED_MODEL and QDRANT_SPARSE_MODEL. "
            f"Received EMBEDDING_PROVIDER={provider.value!r}. "
            "Fix: set EMBEDDING_PROVIDER=fastembed, then verify "
            "QDRANT_EMBED_MODEL=BAAI/bge-small-en-v1.5 and "
            "QDRANT_SPARSE_MODEL=prithivida/Splade_PP_en_v1 are configured."
        )

    return resolved_backend


def validate_indexing_runtime_compatibility(
    backend: IndexerBackend | str | None = None,
    *,
    overwrite: bool = False,
    db: "Database | None" = None,
) -> IndexingCompatibilityResult:
    """Validate indexer runtime compatibility before expensive ingestion work."""
    resolved_backend = _validate_backend_embedding_compatibility(backend)

    if resolved_backend == IndexerBackend.QDRANT:
        from rag_core.core.config import get_qdrant_settings

        get_qdrant_settings()
        return IndexingCompatibilityResult(backend=resolved_backend)

    if resolved_backend == IndexerBackend.POSTGRES:
        from rag_core.core.config import get_embedding_settings

        embedding_settings = get_embedding_settings()
        return _validate_postgres_vector_dimensions(
            expected_dimension=embedding_settings.dimension,
            overwrite=overwrite,
            db=db,
        )

    return IndexingCompatibilityResult(backend=resolved_backend)


def _validate_postgres_vector_dimensions(
    *,
    expected_dimension: int,
    overwrite: bool,
    db: "Database | None" = None,
) -> IndexingCompatibilityResult:
    dimensions = tuple(sorted(_fetch_existing_postgres_vector_dimensions(db=db)))
    result = IndexingCompatibilityResult(
        backend=IndexerBackend.POSTGRES,
        configured_dimension=expected_dimension,
        existing_dimensions=dimensions,
    )
    if not dimensions:
        return result

    if len(dimensions) == 1 and dimensions[0] == expected_dimension:
        return result

    if overwrite:
        return IndexingCompatibilityResult(
            backend=IndexerBackend.POSTGRES,
            configured_dimension=expected_dimension,
            existing_dimensions=dimensions,
            overwrite_rebuild=True,
        )

    formatted_dimensions = ", ".join(str(dim) for dim in dimensions)
    raise ValueError(
        "Dimension mismatch detected. Existing Postgres vector dimension(s): "
        f"{formatted_dimensions}. Configured EMBEDDING_DIMENSION: "
        f"{expected_dimension}. To resolve this and re-index with the new model, "
        "run the ingestion entrypoint once with INGESTION_OVERWRITE=true or call "
        "the pipeline with overwrite=True. Note: This will delete all existing "
        "vectors for this backend."
    )


def _fetch_existing_postgres_vector_dimensions(
    *,
    db: "Database | None" = None,
) -> set[int]:
    if db is None:
        from rag_core.store.postgres.session import get_database

        db = get_database()

    with db.get_session() as session:
        result = session.execute(_get_postgres_dimension_preflight_query())
        return {
            int(dimension)
            for dimension in result.scalars().all()
            if dimension is not None
        }


def _get_postgres_dimension_preflight_query():
    from rag_core.store.postgres.models.documents import DocumentChunk

    table_name = DocumentChunk.__tablename__
    vector_column = DocumentChunk.__table__.c.embedding.name
    return text(
        f"""
        SELECT DISTINCT vector_dims({vector_column}) AS dimension
        FROM {table_name}
        WHERE {vector_column} IS NOT NULL
        LIMIT 2
        """
    )


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
    backend = _validate_backend_embedding_compatibility(backend)

    if backend == IndexerBackend.QDRANT:
        from rag_core.indexing.qdrant_indexer import QdrantIndexer

        logger.info("Indexer backend resolved: Qdrant")
        return QdrantIndexer()

    if backend == IndexerBackend.POSTGRES:
        from rag_core.indexing.postgres_indexer import PostgresIndexer

        logger.info("Indexer backend resolved: Postgres")
        return PostgresIndexer()

    # Unreachable when using the enum, but guards against future additions.
    raise ValueError(f"Unknown indexer backend: '{backend}'")
