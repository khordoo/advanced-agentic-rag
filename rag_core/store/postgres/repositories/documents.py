from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import delete
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
