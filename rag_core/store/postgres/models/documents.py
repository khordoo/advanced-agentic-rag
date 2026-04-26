from __future__ import annotations

import datetime
import enum
from typing import Optional, TYPE_CHECKING

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime,
    Enum as SQLEnum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from rag_core.store.postgres.models.base import Base

if TYPE_CHECKING:
    from rag_core.store.postgres.models.pipelines import PipelineRun


class AssetStatus(str, enum.Enum):
    """Latest processing state for a source asset."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DELETED = "deleted"
    IGNORED = "ignored"


class SourceAsset(Base):
    __tablename__ = "source_assets"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    file_id: Mapped[str] = mapped_column(String(255), unique=True, index=True)

    folder_name: Mapped[str] = mapped_column(String(255))
    file_name: Mapped[str] = mapped_column(String(255))
    source_uri: Mapped[str] = mapped_column(String(4096))
    content_hash: Mapped[str] = mapped_column(String(64))

    status: Mapped[AssetStatus] = mapped_column(
        SQLEnum(AssetStatus, name="asset_status_enum",native_enum=False,length=50),
        default=AssetStatus.PENDING,
        nullable=False,
    )
    worker_id: Mapped[Optional[str]] = mapped_column(String(50))
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    asset_metadata: Mapped[dict] = mapped_column(
        JSONB, server_default="{}", nullable=False
    )

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    last_seen_run_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("pipeline_runs.id", ondelete="SET NULL"), index=True
    )

    last_seen_run: Mapped[Optional["PipelineRun"]] = relationship()

    __table_args__ = (
        UniqueConstraint(
            "folder_name", "file_name", name="uix_source_asset_folder_file"
        ),
    )


class DocumentSection(Base):
    __tablename__ = "document_sections"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    asset_id: Mapped[int] = mapped_column(
        ForeignKey("source_assets.id", ondelete="CASCADE"), index=True
    )

    file_id: Mapped[str] = mapped_column(String(255), index=True)
    source_uri: Mapped[str] = mapped_column(String(4096))
    doc_type: Mapped[str] = mapped_column(String(50))
    page_numbers: Mapped[list[int]] = mapped_column(
        ARRAY(Integer), server_default="{}", nullable=False
    )
    text: Mapped[str] = mapped_column(Text)

    asset: Mapped["SourceAsset"] = relationship()


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    section_id: Mapped[int] = mapped_column(
        ForeignKey("document_sections.id", ondelete="CASCADE"), index=True
    )

    file_id: Mapped[str] = mapped_column(String(255), index=True)
    page_numbers: Mapped[list[int]] = mapped_column(
        ARRAY(Integer), server_default="{}", nullable=False
    )
    text: Mapped[str] = mapped_column(Text)
    vector: Mapped[list[float]] = mapped_column(Vector())

    scopes: Mapped[list[str]] = mapped_column(
        ARRAY(String), server_default="{}", nullable=False
    )
    tags: Mapped[list[str]] = mapped_column(
        ARRAY(String), server_default="{}", nullable=False
    )
    chunk_metadata: Mapped[dict] = mapped_column(
        JSONB, server_default="{}", nullable=False
    )

    section: Mapped["DocumentSection"] = relationship()

    __table_args__ = (
        Index("ix_document_chunks_scopes_gin", "scopes", postgresql_using="gin"),
        Index("ix_document_chunks_tags_gin", "tags", postgresql_using="gin"),
    )
