from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from rag_core.core.config import RetrievalSettings, get_retrieval_settings
from rag_core.embeddings import BaseEmbeddingModel, create_embedding_model
from rag_core.retrieval.base import BaseRetriever
from rag_core.schemas import VectorDatabasePayloadKeys as PayloadKeys
from rag_core.store.postgres.models.documents import DocumentSection
from rag_core.store.postgres.repositories.documents import DocumentRepository
from rag_core.store.postgres.session import Database, get_database


class PostgresRetriever(BaseRetriever):
    def __init__(
        self,
        settings: RetrievalSettings | None = None,
        db: Database | None = None,
        embedding_model: BaseEmbeddingModel | None = None,
    ) -> None:
        self.retrieval_settings = settings or get_retrieval_settings()
        self._db = db or get_database()
        self._embedding_model = embedding_model or create_embedding_model()

    def search(
        self,
        query: str,
        asset_key: str | None = None,
        scopes: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        retrieval_top_k: int | None = None,
        hybrid_alpha: float | None = None,
        rerank_top_k: int | None = None,
        rerank: bool = True,
    ) -> list[Document]:
        del rerank_top_k, rerank
        top_k = retrieval_top_k or self.retrieval_settings.top_k
        fetch_limit = top_k * self.retrieval_settings.oversample_factor
        alpha = hybrid_alpha if hybrid_alpha is not None else self.retrieval_settings.hybrid_alpha
        dense_weight = alpha
        text_weight = 1.0 - alpha

        with self._db.get_session() as session:
            repo = DocumentRepository(session)
            query_embedding = self._embedding_model.embed_query(query)
            dense_chunks = repo.search_chunks_dense(
                query_embedding=query_embedding,
                limit=fetch_limit,
                asset_key=asset_key,
                scopes=scopes,
                exact_filters=filters,
            )
            fts_chunks = repo.search_chunks_fts(
                query=query,
                limit=fetch_limit,
                asset_key=asset_key,
                scopes=scopes,
                exact_filters=filters,
            )

            dense_ranks = {int(chunk.id): rank for rank, chunk in enumerate(dense_chunks, start=1)}
            fts_ranks = {int(chunk.id): rank for rank, chunk in enumerate(fts_chunks, start=1)}
            fused_scores = self._fuse_rank_maps(
                dense_ranks=dense_ranks,
                fts_ranks=fts_ranks,
                dense_weight=dense_weight,
                text_weight=text_weight,
                rrf_k=self.retrieval_settings.postgres_rrf_k,
            )

            if not fused_scores:
                return []

            chunk_by_id = {int(chunk.id): chunk for chunk in [*dense_chunks, *fts_chunks]}
            section_score_map: dict[int, float] = {}
            for chunk_id, score in fused_scores.items():
                chunk = chunk_by_id.get(chunk_id)
                if chunk is None or chunk.section_id is None:
                    continue
                section_score_map[chunk.section_id] = max(
                    section_score_map.get(chunk.section_id, 0.0), score
                )

            ranked_section_ids = [
                section_id
                for section_id, _ in sorted(
                    section_score_map.items(), key=lambda item: item[1], reverse=True
                )[:top_k]
            ]
            sections = repo.fetch_sections_by_ids(section_ids=ranked_section_ids)
            section_lookup = {int(section.id): section for section in sections}

            documents: list[Document] = []
            for section_id in ranked_section_ids:
                section = section_lookup.get(section_id)
                if section is None:
                    continue
                documents.append(
                    self._section_to_document(
                        section=section,
                        score=section_score_map[section_id],
                    )
                )

            return documents

    def exact_lookup(
        self,
        asset_key: str | None = None,
        filters: dict[str, Any] | None = None,
        scopes: list[str] | None = None,
        limit: int | None = None,
    ) -> list[Document]:
        top_k = limit or self.retrieval_settings.top_k
        with self._db.get_session() as session:
            repo = DocumentRepository(session)
            sections = repo.lookup_sections(
                asset_key=asset_key,
                scopes=scopes,
                exact_filters=filters,
                limit=top_k,
            )
            return [self._section_to_document(section=section) for section in sections]

    @staticmethod
    def _resolve_asset_name(section: DocumentSection) -> str:
        asset = getattr(section, "asset", None)
        if asset is not None:
            name = getattr(asset, "asset_name", "")
            if name:
                return name
        return ""

    @classmethod
    def _section_to_document(
        cls,
        *,
        section: DocumentSection,
        score: float | None = None,
    ) -> Document:
        metadata = {
            PayloadKeys.ASSET_KEY.value: section.asset_key,
            PayloadKeys.ASSET_NAME.value: cls._resolve_asset_name(section),
            PayloadKeys.SOURCE.value: section.source_uri,
            PayloadKeys.PAGE.value: section.page_numbers,
            "id": section.id,
        }
        if section.created_at:
            metadata[PayloadKeys.CREATED_AT.value] = section.created_at.isoformat()
        if score is not None:
            metadata["score"] = score
        return Document(page_content=section.text, metadata=metadata)

    @staticmethod
    def _fuse_rank_maps(
        *,
        dense_ranks: dict[int, int],
        fts_ranks: dict[int, int],
        dense_weight: float,
        text_weight: float,
        rrf_k: int,
    ) -> dict[int, float]:
        all_chunk_ids = set(dense_ranks).union(set(fts_ranks))
        fused_scores: dict[int, float] = {}

        for chunk_id in all_chunk_ids:
            d_score = 1.0 / (dense_ranks[chunk_id] + rrf_k) if chunk_id in dense_ranks else 0.0
            f_score = 1.0 / (fts_ranks[chunk_id] + rrf_k) if chunk_id in fts_ranks else 0.0
            fused_scores[chunk_id] = (dense_weight * d_score) + (text_weight * f_score)

        return fused_scores
