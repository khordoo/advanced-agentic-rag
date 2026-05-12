from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from pgvector.sqlalchemy import Vector
from sqlalchemy import Computed, DateTime, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column, relationship

from rag_core.core.config import get_ingestion_settings
from rag_core.store.postgres.models.base import Base

if TYPE_CHECKING:
    from rag_core.store.postgres.models.ingestion import SourceAsset

settings = get_ingestion_settings()


class DocumentSection(Base):
    __tablename__ = "document_sections"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    asset_id: Mapped[int] = mapped_column(
        ForeignKey("source_assets.id", ondelete="CASCADE"), index=True
    )

    asset_key: Mapped[str] = mapped_column(String(255), index=True)
    source_uri: Mapped[str] = mapped_column(String(4096))
    doc_type: Mapped[str] = mapped_column(String(50))
    page_numbers: Mapped[list[int]] = mapped_column(
        ARRAY(Integer), server_default="{}", nullable=False
    )
    text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    asset: Mapped["SourceAsset"] = relationship()

    def __repr__(self) -> str:
        return (
            "DocumentSection("
            f"id={self.id}, "
            f"asset_id={self.asset_id}, "
            f"asset_key={self.asset_key!r}, "
            f"page_numbers={self.page_numbers}"
            ")"
        )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    section_id: Mapped[int] = mapped_column(
        ForeignKey("document_sections.id", ondelete="CASCADE"), index=True
    )

    asset_key: Mapped[str] = mapped_column(String(255), index=True)
    page_numbers: Mapped[list[int]] = mapped_column(
        ARRAY(Integer), server_default="{}", nullable=False
    )
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector())
    fts_vector: Mapped[list[float]] = mapped_column(
        TSVECTOR, Computed(f"to_tsvector('{settings.fts_language}', text)", persisted=True)
    )

    scopes: Mapped[list[str]] = mapped_column(ARRAY(String), server_default="{}", nullable=False)
    tags: Mapped[list[str]] = mapped_column(ARRAY(String), server_default="{}", nullable=False)
    chunk_metadata: Mapped[dict] = mapped_column(JSONB, server_default="{}", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    section: Mapped["DocumentSection"] = relationship()

    __table_args__ = (
        Index("ix_document_chunks_scopes_gin", "scopes", postgresql_using="gin"),
        Index("ix_document_chunks_tags_gin", "tags", postgresql_using="gin"),
        Index("ix_document_keyword_search_gin", "fts_vector", postgresql_using="gin"),
    )

    def __repr__(self) -> str:
        return (
            "DocumentChunk("
            f"id={self.id}, "
            f"section_id={self.section_id}, "
            f"asset_key={self.asset_key!r}, "
            f"page_numbers={self.page_numbers}, "
            f"scopes={self.scopes}, "
            f"tags={self.tags}, "
            f"embedding=[{self.embedding[:3]}...]"
            ")"
        )
