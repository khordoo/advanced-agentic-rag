"""pgvector-backed indexer for parent–child document storage.

Persists :class:`~rag_core.schemas.ParentChunk` trees into Postgres using
the existing :class:`~rag_core.store.postgres.models.documents.DocumentSection`
and :class:`~rag_core.store.postgres.models.documents.DocumentChunk` tables.

Embeddings are generated via **fastembed** — the same model used by the Qdrant
indexer — so vector representations are identical regardless of which backend
is selected.
"""

from __future__ import annotations

import logging
from typing import Any

from rag_core.core.config import (
    IngestionSettings,
    QdrantSettings,
    get_ingestion_settings,
    get_qdrant_settings,
)
from rag_core.indexing.base import BaseIndexer
from rag_core.schemas import ParentChunk
from rag_core.store.postgres.models.documents import DocumentChunk, DocumentSection
from rag_core.store.postgres.session import Database, get_database

logger = logging.getLogger(__name__)


def _get_fastembed_model(model_name: str):
    """Lazy-load a fastembed TextEmbedding model."""
    from fastembed import TextEmbedding  # type: ignore[import-untyped]

    return TextEmbedding(model_name=model_name)


class PgVectorIndexer(BaseIndexer):
    """Indexes :class:`ParentChunk` trees into Postgres with pgvector.

    Parameters
    ----------
    db:
        Database wrapper for session management.  Falls back to the
        process-wide singleton.
    settings:
        Ingestion settings providing ``embedding_dimension``.
    qdrant_settings:
        Qdrant settings providing the fastembed model name.  We reuse this
        so both backends always embed with the same model.
    """

    def __init__(
        self,
        db: Database | None = None,
        settings: IngestionSettings | None = None,
        qdrant_settings: QdrantSettings | None = None,
    ) -> None:
        self._db = db or get_database()
        self._settings = settings or get_ingestion_settings()
        self._qdrant_settings = qdrant_settings or get_qdrant_settings()

        self._embed_model_name = self._qdrant_settings.embed_model
        self._embedding_dimension = self._settings.embedding_dimension

        # Lazy-initialised on first use so import cost is deferred
        self._embed_model = None

        logger.info(
            "PgVectorIndexer initialised",
            extra={
                "embed_model": self._embed_model_name,
                "embedding_dimension": self._embedding_dimension,
            },
        )

    @property
    def embed_model(self):
        """Lazy-load the fastembed model on first access."""
        if self._embed_model is None:
            logger.info(
                "Loading fastembed model '%s' for pgvector …",
                self._embed_model_name,
            )
            self._embed_model = _get_fastembed_model(self._embed_model_name)
        return self._embed_model

    # ------------------------------------------------------------------
    # BaseIndexer contract
    # ------------------------------------------------------------------

    def ingest_documents(self, document_blocks: list[ParentChunk]) -> None:
        """Persist parent + child chunks into Postgres.

        Each :class:`ParentChunk` produces:
        * One :class:`DocumentSection` row (the parent / LLM payload).
        * Multiple :class:`DocumentChunk` rows (embedded child chunks).
        """
        if not document_blocks:
            logger.info("No document blocks to ingest — skipping.")
            return

        logger.info(
            "Preparing %d document blocks for pgvector ingestion …",
            len(document_blocks),
        )

        # Collect all child texts for batch embedding
        all_child_texts: list[str] = []
        child_index_map: list[tuple[int, int]] = []  # (block_idx, child_idx)

        for block_idx, block in enumerate(document_blocks):
            for child_idx, child in enumerate(block.children):
                all_child_texts.append(child.page_content)
                child_index_map.append((block_idx, child_idx))

        # Batch embed
        vectors = self._embed_texts(all_child_texts) if all_child_texts else []

        # Build a lookup: (block_idx, child_idx) → vector
        vector_lookup: dict[tuple[int, int], list[float]] = {}
        for (block_idx, child_idx), vec in zip(child_index_map, vectors):
            vector_lookup[(block_idx, child_idx)] = vec

        # Write to Postgres
        with self._db.get_session() as session:
            with session.begin():
                for block_idx, block in enumerate(document_blocks):
                    section = self._build_section(block)
                    session.add(section)
                    session.flush()  # get section.id

                    for child_idx, child in enumerate(block.children):
                        vec = vector_lookup.get((block_idx, child_idx), [])
                        chunk = self._build_chunk(
                            section=section,
                            child=child,
                            block=block,
                            vector=vec,
                        )
                        session.add(chunk)

        logger.info(
            "pgvector ingestion complete: %d sections, %d chunks.",
            len(document_blocks),
            len(all_child_texts),
        )

    def reset_collections(self) -> None:
        """Truncate document_chunks and document_sections tables.

        Uses TRUNCATE CASCADE so foreign-key dependents are cleaned up
        atomically.
        """
        logger.info("Resetting pgvector document tables …")
        with self._db.get_session() as session:
            with session.begin():
                session.execute(
                    DocumentChunk.__table__.delete()
                )
                session.execute(
                    DocumentSection.__table__.delete()
                )
        logger.info("pgvector document tables truncated.")

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using the fastembed model."""
        logger.info("Embedding %d texts via fastembed …", len(texts))
        # fastembed returns a generator of numpy arrays
        embeddings = list(self.embed_model.embed(texts))
        result = [emb.tolist() for emb in embeddings]

        if result and len(result[0]) != self._embedding_dimension:
            logger.warning(
                "Embedding dimension mismatch: model produces %d but config expects %d. "
                "Update INGESTION_EMBEDDING_DIMENSION to match your model.",
                len(result[0]),
                self._embedding_dimension,
            )
        return result

    # ------------------------------------------------------------------
    # Row construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_section(block: ParentChunk) -> DocumentSection:
        """Build a DocumentSection row from a ParentChunk."""
        if block.asset_id is None:
            raise ValueError(
                f"ParentChunk '{block.id}' is missing asset_id. "
                "The processor must stamp each chunk with the source asset PK."
            )
        meta = block.metadata
        return DocumentSection(
            asset_id=block.asset_id,
            file_id=meta.file_name,
            source_uri=meta.source,
            doc_type="table" if meta.table_id else "text",
            page_numbers=meta.pages,
            text=block.enriched_text,
        )

    @staticmethod
    def _build_chunk(
        section: DocumentSection,
        child: Any,
        block: ParentChunk,
        vector: list[float],
    ) -> DocumentChunk:
        """Build a DocumentChunk row from a child document."""
        meta = child.metadata if hasattr(child, "metadata") else {}
        return DocumentChunk(
            section_id=section.id,
            file_id=meta.get("file_name", block.metadata.file_name),
            page_numbers=meta.get("pages", block.metadata.pages),
            text=child.page_content,
            vector=vector,
            scopes=meta.get("scopes", block.metadata.scopes),
            tags=meta.get("tags", block.metadata.tags),
            chunk_metadata={
                k: v
                for k, v in meta.items()
                if k not in {"file_name", "pages", "scopes", "tags"}
            },
        )
