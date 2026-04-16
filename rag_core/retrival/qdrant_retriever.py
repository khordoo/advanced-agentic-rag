"""Qdrant-backed retriever for hierarchical search.

Fetches child chunks based on semantic similarity, reduces them to unique parents,
and retrieves the full parent documents in a single batch.
"""
from __future__ import annotations

import logging

from typing import Any

from langchain_core.documents import Document
from qdrant_client import QdrantClient, models

from rag_core.config import RetrivalSettings, QdrantSettings, get_qdrant_settings
from rag_core.db.connection_manager import QdrantConnectionManager
from rag_core.retrival.base import BaseRetriever
from rag_core.schemas import VectorDatabasePayloadKeys as PayloadKeys, PARENT_INDEX_SCHEMA

logger = logging.getLogger(__name__)


class QdrantRetriever(BaseRetriever):
    """Hierarchical retriever using Qdrant.

    Implements a two-step retrieval process:
    1. Search child chunks for the best matches.
    2. Retrieve corresponding parent documents for the LLM payload.
    """

    def __init__(
        self,
        client: QdrantClient | None = None,
        settings: QdrantSettings | None = None,
        retrieval_settings: RetrivalSettings | None = None,
    ) -> None:
        super().__init__()
        self._client = client or QdrantConnectionManager.get_client()
        self._settings = settings or get_qdrant_settings()
        self.retrieval_settings = retrieval_settings or RetrivalSettings()

    def exact_lookup(
        self,
        scopes: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[Document]:
        """Perform a deterministic metadata lookup."""
        limit= limit if limit is None else self.retrieval_settings.top_k
        qdrant_filter = self._build_filters(scopes, exact_filters=filters)

        points, _ = self._client.scroll(
            collection_name=self._settings.parent_collection_name,
            scroll_filter=qdrant_filter,
            limit=limit,
            with_payload=True,
        )
        return [self._point_to_document(p) for p in points]



    def _point_to_document(self, point: models.Record | models.ScoredPoint) -> Document:
        """Convert a Qdrant point back to a LangChain Document."""
        payload = point.payload or {}
        text = payload.get(PayloadKeys.TEXT.value, "")

        # Only include metadata fields defined in the schema
        metadata = {k: v for k, v in payload.items() if k in PARENT_INDEX_SCHEMA}
        metadata["id"] = point.id

        return Document(page_content=text, metadata=metadata)


    def search(
        self,
        query: str,
        limit: int | None = None,
        score_threshold: float | None = None,
        scopes: list[str] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Execute hybrid search (dense + sparse RRF) with parent document reconstruction."""

        actual_limit = limit if limit is not None else self.retrieval_settings.top_k
        actual_threshold = score_threshold if score_threshold is not None else self.retrieval_settings.rrf_score_threshold

        qdrant_filter = self._build_filters(scopes, exact_filters=filters)
        fetch_limit = actual_limit * self.retrieval_settings.oversample_factor

        # Embed query for both dense and sparse search
        dense_name, *_ = self._client.get_fastembed_vector_params()
        sparse_name, *_ = self._client.get_fastembed_sparse_vector_params()

        (_, dense_vector), *_ = self._client._embed_documents(
            documents=[query],
            embedding_model_name=self._client.embedding_model_name,
        )
        sparse_vector = next(self._client._sparse_embed_documents(
            documents=[query],
            embedding_model_name=self._client.sparse_embedding_model_name,
        ))

        # Hybrid search: prefetch dense + sparse, fuse with RRF
        response = self._client.query_points(
            collection_name=self._settings.chunk_collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using=dense_name,
                    filter=qdrant_filter,
                    limit=fetch_limit,
                ),
                models.Prefetch(
                    query=sparse_vector,
                    using=sparse_name,
                    filter=qdrant_filter,
                    limit=fetch_limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=fetch_limit,
        )
        results = response.points

        # Extract unique parents and their maximum scores
        unique_parents = self._extract_unique_parents(results=results,
                                                      limit=actual_limit,
                                                      score_threshold= actual_threshold)
        if not unique_parents:
            return []
        logger.info(f"Processed {len(results)} -> {len(unique_parents)} unique parent documents.")
        # Batch retrieve parent documents
        parent_ids = [pid for pid, _ in unique_parents]
        score_map = dict(unique_parents)
        parent_points = self._client.retrieve(
            collection_name=self._settings.parent_collection_name,
            ids=parent_ids,
            with_payload=True,
        )
        # Reconstruct documents and sort by retrieval score
        documents = []
        for point in parent_points:
            doc = self._point_to_document(point)
            # Map score from child back to parent
            normalized_parent_id = str(point.id).replace('-', '')
            doc.metadata["retrieval_score"] = score_map.get(normalized_parent_id, 0.0)
            documents.append(doc)

        documents.sort(key=lambda d: d.metadata["retrieval_score"], reverse=True)
        return documents

    def _build_filters(
        self,
        scopes: list[str] | None,
        exact_filters: dict[str, Any] | None,
    ) -> models.Filter:
        """Construct Qdrant Filter from scopes and exact metadata matches."""

        must_conditions: list[models.Condition] =[]

        if scopes is not None:
            must_conditions.append(
                models.FieldCondition(
                    key=PayloadKeys.SCOPES.value,
                    match=models.MatchAny(any=scopes),
                )
        )

        if exact_filters:
            valid_keys = {k.value for k in PayloadKeys if k != PayloadKeys.TEXT}
            for key, val in exact_filters.items():
                if key not in valid_keys:
                    raise KeyError(f"Invalid filter key: {key}. Allowed: {valid_keys}")

                match = models.MatchAny(any=val) if isinstance(val, list) else models.MatchValue(value=val)
                must_conditions.append(models.FieldCondition(key=key, match=match))

        return models.Filter(must=must_conditions)

    def _extract_unique_parents(
        self,
        results: list[models.ScoredPoint],
        limit: int,
        score_threshold:int,

    ) -> list[tuple[str, float]]:
        """Group child chunks by parent_id, keeping the highest score for each."""
        unique_ids: dict[str, float] = {}
        for res in results:
            if res.score < score_threshold:
                continue
            payload = res.payload or {}
            pid = payload.get(PayloadKeys.PARENT_ID.value, None)
            if not pid:
                continue

            unique_ids[pid] = max(unique_ids.get(pid, 0.0), res.score)

        sorted_ids = sorted(unique_ids.items(), key=lambda x: x[1], reverse=True)
        print('sorted_ids=',sorted_ids)
        return sorted_ids[: limit]

    def _point_to_document(self, point: models.Record | models.ScoredPoint) -> Document:
        """Convert a Qdrant point back to a LangChain Document."""
        payload = point.payload or {}
        text = payload.get(PayloadKeys.TEXT.value, "")
        
        # Only include metadata fields defined in the schema
        metadata = {k: v for k, v in payload.items() if k in PARENT_INDEX_SCHEMA}
        metadata["id"] = point.id

        return Document(page_content=text, metadata=metadata)
