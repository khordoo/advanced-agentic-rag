"""Postgres-backed indexer for parent-child document storage.

Persists :class:`~rag_core.schemas.ParentChunk` trees into Postgres using
the existing :class:`~rag_core.store.postgres.models.documents.DocumentSection`
and :class:`~rag_core.store.postgres.models.documents.DocumentChunk` tables.

Embeddings are provided through the shared embedding model abstraction so the
provider can be changed without touching database indexing logic.
"""

from __future__ import annotations

import logging
from typing import Any

from rag_core.core.config import IngestionSettings
from rag_core.embeddings import BaseEmbeddingModel, create_embedding_model
from rag_core.indexing.base import BaseIndexer
from rag_core.schemas import ClaimedSourceAsset, ParentChunk
from rag_core.store.postgres.models.documents import DocumentChunk, DocumentSection
from rag_core.store.postgres.repositories.documents import DocumentRepository
from rag_core.store.postgres.session import Database, get_database

logger = logging.getLogger(__name__)


class PostgresIndexer(BaseIndexer):
    """Indexes :class:`ParentChunk` trees into Postgres document tables.

    Parameters
    ----------
    db:
        Database wrapper for session management.  Falls back to the
        process-wide singleton.
    settings:
        Optional ingestion settings retained for caller compatibility.
    embedding_model:
        Dense embedding provider used to embed child chunks.
    """

    def __init__(
        self,
        db: Database | None = None,
        settings: IngestionSettings | None = None,
        embedding_model: BaseEmbeddingModel | None = None,
    ) -> None:
        self._db = db or get_database()
        self._settings = settings
        self._embedding_model = embedding_model or create_embedding_model()
        self._embedding_dimension = self._embedding_model.dimension

        logger.info(
            "PostgresIndexer initialised",
            extra={
                "embedding_model": type(self._embedding_model).__name__,
                "embedding_dimension": self._embedding_dimension,
            },
        )

    # ------------------------------------------------------------------
    # BaseIndexer contract
    # ------------------------------------------------------------------

    def index_asset(
        self,
        asset: ClaimedSourceAsset,
        document_blocks: list[ParentChunk],
    ) -> None:
        """Atomically replace the Postgres index for one claimed source asset."""
        self._validate_blocks_for_asset(asset.id, document_blocks)
        self._sync_document_blocks(asset.id, document_blocks)

    def _sync_document_blocks(
        self,
        asset_id: int,
        document_blocks: list[ParentChunk],
    ) -> None:
        """Build fresh rows and replace all indexed rows for one asset."""
        logger.info(
            "Preparing %d document blocks for Postgres asset sync ...",
            len(document_blocks),
        )

        sections, chunks = self._build_index_rows(document_blocks)

        with self._db.get_session() as session:
            with session.begin():
                repo = DocumentRepository(session)
                deleted_sections = repo.sync_document_index(
                    asset_id=asset_id,
                    sections=sections,
                    chunks=chunks,
                )

        logger.info(
            "Postgres asset sync complete: asset_id=%d, deleted_sections=%d, "
            "%d sections, %d chunks.",
            asset_id,
            deleted_sections,
            len(sections),
            len(chunks),
        )

    def _build_index_rows(
        self,
        document_blocks: list[ParentChunk],
    ) -> tuple[list[DocumentSection], list[DocumentChunk]]:
        """Build unsaved section and chunk rows for a validated asset batch."""
        all_child_texts: list[str] = []
        child_index_map: list[tuple[int, int]] = []  # (block_idx, child_idx)

        for block_idx, block in enumerate(document_blocks):
            for child_idx, child in enumerate(block.children):
                all_child_texts.append(child.page_content)
                child_index_map.append((block_idx, child_idx))

        vectors = self._embed_texts(all_child_texts) if all_child_texts else []

        vector_lookup: dict[tuple[int, int], list[float]] = {}
        for (block_idx, child_idx), vec in zip(child_index_map, vectors):
            vector_lookup[(block_idx, child_idx)] = vec

        sections: list[DocumentSection] = []
        chunks: list[DocumentChunk] = []

        for block_idx, block in enumerate(document_blocks):
            section = self._build_section(block)
            sections.append(section)

            for child_idx, child in enumerate(block.children):
                vec = vector_lookup.get((block_idx, child_idx), [])
                chunks.append(
                    self._build_chunk(
                        section=section,
                        child=child,
                        block=block,
                        embedding=vec,
                    )
                )

        return sections, chunks

    def reset_collections(self) -> None:
        """Clear the Postgres document tables.

        Uses ORM table metadata so model table-name changes are reflected
        automatically.
        """
        logger.info("Resetting Postgres document tables ...")
        with self._db.get_session() as session:
            with session.begin():
                session.execute(DocumentChunk.__table__.delete())
                session.execute(DocumentSection.__table__.delete())
        logger.info("Postgres document tables truncated.")

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts and validate provider output."""
        logger.info("Embedding %d texts for Postgres indexing ...", len(texts))
        vectors = self._embedding_model.embed_documents(texts)
        if len(vectors) != len(texts):
            raise ValueError(
                "Embedding model returned "
                f"{len(vectors)} vectors for {len(texts)} input texts."
            )
        self._validate_embedding_vectors(vectors)
        return vectors

    def _validate_embedding_vectors(self, vectors: list[list[float]]) -> None:
        for index, vector in enumerate(vectors):
            if len(vector) != self._embedding_dimension:
                raise ValueError(
                    "Embedding dimension mismatch at index "
                    f"{index}: model produced {len(vector)} values but "
                    f"configuration expects {self._embedding_dimension}."
                )

    # ------------------------------------------------------------------
    # Row construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_blocks_for_asset(
        asset_id: int,
        document_blocks: list[ParentChunk],
    ) -> None:
        """Ensure every processed block belongs to the claimed asset."""
        for block in document_blocks:
            if block.asset_id is None:
                raise ValueError(
                    f"ParentChunk '{block.id}' is missing asset_id. "
                    "The processor must stamp each chunk with the source asset PK."
                )
            if block.asset_id != asset_id:
                raise ValueError(
                    f"ParentChunk '{block.id}' belongs to asset_id={block.asset_id}, "
                    f"but the claimed asset is asset_id={asset_id}."
                )

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
            asset_key=meta.asset_key,
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
        embedding: list[float],
    ) -> DocumentChunk:
        """Build a DocumentChunk row from a child document."""
        meta = dict(child.metadata if hasattr(child, "metadata") else {})
        if section.id is not None:
            meta["parent_id"] = section.id

        fields: dict[str, Any] = {
            "section": section,
            "asset_key": meta.get("asset_key", block.metadata.asset_key),
            "page_numbers": meta.get("pages", block.metadata.pages),
            "text": child.page_content,
            "embedding": embedding,
            "scopes": meta.get("scopes", block.metadata.scopes),
            "tags": meta.get("tags", block.metadata.tags),
            "chunk_metadata": {
                k: v
                for k, v in meta.items()
                if k not in {"asset_key", "file_name", "pages", "scopes", "tags"}
            },
        }
        if section.id is not None:
            fields["section_id"] = section.id
        return DocumentChunk(**fields)
