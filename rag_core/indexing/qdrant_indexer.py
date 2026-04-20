"""Qdrant-backed indexer for parent–child document storage.

Manages two collections per namespace:
  • ``{namespace}_chunks``  – child chunks with dense + sparse vectors
  • ``{namespace}_parents`` – parent documents stored as payload-only points
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient, models

from rag_core.core.config import QdrantSettings, get_qdrant_settings
from rag_core.db.connection_manager import QdrantConnectionManager
from rag_core.indexing.base import BaseIndexer
from rag_core.schemas import (
    CHILD_INDEX_SCHEMA,
    PARENT_INDEX_SCHEMA,
    ParentChunk,
    VectorDatabasePayloadKeys as PayloadKeys,
)

logger = logging.getLogger(__name__)


class QdrantIndexer(BaseIndexer):
    """Indexes :class:`ParentChunk` trees into a pair of Qdrant collections.

    Parameters
    ----------
    client:
        Pre-configured ``QdrantClient``.  When *None*, one is obtained via
        :class:`QdrantConnectionManager`.
    settings:
        Qdrant-specific configuration.  Falls back to :func:`get_qdrant_settings`.
    """

    def __init__(
        self,
        client: QdrantClient | None = None,
        settings: QdrantSettings | None = None,
    ) -> None:
        self._client = client or QdrantConnectionManager.get_client()
        self._settings = settings or get_qdrant_settings()

        self._chunk_collection = self._settings.chunk_collection_name
        self._parent_collection = self._settings.parent_collection_name

        self._ensure_collections()

    # ------------------------------------------------------------------
    # Collection lifecycle
    # ------------------------------------------------------------------

    def _ensure_collections(self) -> None:
        """Ensure both collections exist with the correct schema."""
        self._create_collection_if_missing(
            self._chunk_collection,
            CHILD_INDEX_SCHEMA,
            vector_enabled=True,
        )
        self._create_collection_if_missing(
            self._parent_collection,
            PARENT_INDEX_SCHEMA,
            vector_enabled=False,
        )

    def _create_collection_if_missing(
        self,
        name: str,
        schema: dict[str, Any],
        *,
        vector_enabled: bool = True,
    ) -> None:
        """Create a collection and apply payload indexes when it does not exist."""
        if self._client.collection_exists(name):
            return

        logger.info("Creating collection '%s' (vectors=%s) …", name, vector_enabled)

        if vector_enabled:
            self._client.create_collection(
                collection_name=name,
                vectors_config=self._client.get_fastembed_vector_params(),
                sparse_vectors_config=self._client.get_fastembed_sparse_vector_params(),
            )
        else:
            self._client.create_collection(
                collection_name=name,
                vectors_config={},
            )

        for field_name, field_type in schema.items():
            self._client.create_payload_index(
                collection_name=name,
                field_name=field_name,
                field_schema=field_type,
            )
        logger.info("Payload indexes applied for '%s'.", name)

    def reset_collections(self) -> None:
        """Drop and recreate both managed collections."""
        logger.info("Resetting collections …")
        for name in (self._chunk_collection, self._parent_collection):
            self._drop_collection(name)
        self._ensure_collections()

    def ingest_documents(self, document_blocks: list[ParentChunk]) -> None:
        """Upsert parent + child points into Qdrant.

        Each :class:`ParentChunk` produces:
        * One payload-only point in the *parents* collection.
        * Multiple child chunks in the *chunks* collection (embedded via fastembed).
        """
        logger.info(
            "Preparing %d document blocks for ingestion into Qdrant …",
            len(document_blocks),
        )

        parent_points: list[models.PointStruct] = []
        child_texts: list[str] = []
        child_metadatas: list[dict[str, Any]] = []
        child_ids: list[str] = []

        for block in document_blocks:
            parent_points.append(self._build_parent_point(block))

            for child in block.children:
                child_texts.append(child.page_content)
                child_metadatas.append(child.metadata)
                child_ids.append(str(uuid.uuid4()))

        if child_texts:
            child_points = self._build_child_points(
                child_ids, child_texts, child_metadatas
            )
            self._upsert(self._chunk_collection, child_points, label="child chunks")

        if parent_points:
            self._upsert(self._parent_collection, parent_points, label="parents")

        logger.info(
            "Ingestion complete: %d parents, %d child chunks → [%s, %s].",
            len(parent_points),
            len(child_texts),
            self._parent_collection,
            self._chunk_collection,
        )

    # ------------------------------------------------------------------
    # Point construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_parent_point(block: ParentChunk) -> models.PointStruct:
        """Build a vector-less point for the parents collection."""
        return models.PointStruct(
            id=block.id,
            vector={},
            payload={"text": block.enriched_text, **block.metadata.model_dump()},
        )

    def _build_child_points(
        self,
        ids: list[str],
        texts: list[str],
        metadatas: list[dict[str, Any]],
    ) -> list[models.PointStruct]:
        """Embed child chunks via fastembed and return as PointStruct objects."""
        logger.info("Embedding %d child chunks …", len(texts))

        dense_vectors = list(
            self._client._embed_documents(
                documents=texts,
                embedding_model_name=self._client.embedding_model_name,
            ),
        )
        sparse_vectors = list(
            self._client._sparse_embed_documents(
                documents=texts,
                embedding_model_name=self._client.sparse_embedding_model_name,
            ),
        )

        dense_name, *_ = self._client.get_fastembed_vector_params()
        sparse_name, *_ = self._client.get_fastembed_sparse_vector_params()

        return [
            models.PointStruct(
                id=cid,
                vector={dense_name: dense_vec, sparse_name: sparse_vec},
                payload={PayloadKeys.TEXT.value: text, **meta},
            )
            for cid, (_, dense_vec), sparse_vec, text, meta in zip(
                ids,
                dense_vectors,
                sparse_vectors,
                texts,
                metadatas,
            )
        ]

    def _upsert(
        self,
        collection: str,
        points: list[models.PointStruct],
        *,
        label: str,
    ) -> None:
        logger.info("Upserting %d %s into '%s' …", len(points), label, collection)
        self._client.upsert(collection_name=collection, points=points)

    def _drop_collection(self, name: str) -> None:
        if self._client.collection_exists(name):
            self._client.delete_collection(collection_name=name)
            logger.info("Dropped collection '%s'.", name)
        else:
            logger.debug("Collection '%s' does not exist — skipped.", name)
