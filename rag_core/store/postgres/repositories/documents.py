from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import delete, desc, func, select
from sqlalchemy.orm import Session

from rag_core.store.postgres.models.documents import DocumentChunk, DocumentSection


class DocumentRepository:
    """Persistence operations for indexed document rows."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def sync_document_index(
        self,
        asset_id: int,
        sections: Sequence[DocumentSection],
        chunks: Sequence[DocumentChunk],
    ) -> int:
        """Replace all indexed rows for one source asset.

        The caller owns the transaction. Deleting sections is enough because
        document_chunks has an ON DELETE CASCADE foreign key to sections.
        """
        result = self.session.execute(
            delete(DocumentSection).where(DocumentSection.asset_id == asset_id)
        )

        self.session.add_all(sections)
        self.session.flush()

        for chunk in chunks:
            if chunk.section is None or chunk.section.id is None:
                continue
            chunk.section_id = chunk.section.id
            chunk_metadata = dict(chunk.chunk_metadata or {})
            chunk_metadata["parent_id"] = chunk.section.id
            chunk.chunk_metadata = chunk_metadata

        self.session.add_all(chunks)
        return result.rowcount or 0

    def search_chunks_dense(
        self,
        *,
        query_embedding: list[float],
        limit: int,
        asset_key: str | None = None,
        scopes: list[str] | None = None,
        exact_filters: dict | None = None,
    ) -> list[DocumentChunk]:
        distance = DocumentChunk.embedding.cosine_distance(query_embedding).label("distance")
        stmt = select(DocumentChunk, distance).order_by(distance.asc()).limit(limit)
        stmt = self._apply_chunk_filters(
            stmt,
            asset_key=asset_key,
            scopes=scopes,
            exact_filters=exact_filters,
        )
        rows = self.session.execute(stmt).all()
        return [row[0] for row in rows]

    def search_chunks_fts(
        self,
        *,
        query: str,
        limit: int,
        asset_key: str | None = None,
        scopes: list[str] | None = None,
        exact_filters: dict | None = None,
    ) -> list[DocumentChunk]:
        ts_query = func.websearch_to_tsquery("english", query)
        stmt = (
            select(
                DocumentChunk,
                func.ts_rank_cd(DocumentChunk.fts_vector, ts_query).label("keyword_score"),
            )
            .where(DocumentChunk.fts_vector.op("@@")(ts_query))
            .order_by(desc("keyword_score"))
            .limit(limit)
        )
        stmt = self._apply_chunk_filters(
            stmt,
            asset_key=asset_key,
            scopes=scopes,
            exact_filters=exact_filters,
        )
        rows = self.session.execute(stmt).all()
        return [row[0] for row in rows]

    def fetch_sections_by_ids(self, *, section_ids: list[int]) -> list[DocumentSection]:
        if not section_ids:
            return []
        stmt = select(DocumentSection).where(DocumentSection.id.in_(section_ids))
        rows = self.session.execute(stmt).scalars().all()
        return list(rows)

    def lookup_sections(
        self,
        *,
        asset_key: str | None = None,
        scopes: list[str] | None = None,
        exact_filters: dict | None = None,
        limit: int = 10,
    ) -> list[DocumentSection]:
        chunk_stmt = select(DocumentChunk.section_id).limit(limit)
        chunk_stmt = self._apply_chunk_filters(
            chunk_stmt,
            asset_key=asset_key,
            scopes=scopes,
            exact_filters=exact_filters,
        )
        section_ids = self.session.execute(chunk_stmt).scalars().all()
        ordered_section_ids: list[int] = []
        for sid in section_ids:
            if sid is not None and sid not in ordered_section_ids:
                ordered_section_ids.append(sid)
        return self.fetch_sections_by_ids(section_ids=ordered_section_ids[:limit])

    def _apply_chunk_filters(
        self,
        stmt,
        *,
        asset_key: str | None,
        scopes: list[str] | None,
        exact_filters: dict | None,
    ):
        if asset_key is not None:
            stmt = stmt.where(DocumentChunk.asset_key == asset_key)
        if scopes:
            stmt = stmt.where(DocumentChunk.scopes.overlap(scopes))
        if exact_filters:
            for key, value in exact_filters.items():
                column = getattr(DocumentChunk, key, None)
                if column is not None:
                    if isinstance(value, list):
                        if key in {"scopes", "tags", "page_numbers"}:
                            stmt = stmt.where(column.overlap(value))
                        else:
                            stmt = stmt.where(column.in_(value))
                    else:
                        stmt = stmt.where(column == value)
                    continue

                if key in {"page", "pages"}:
                    page_values = value if isinstance(value, list) else [value]
                    stmt = stmt.where(DocumentChunk.page_numbers.overlap(page_values))
                    continue

                stmt = stmt.where(DocumentChunk.chunk_metadata[key].astext == str(value))
        return stmt
