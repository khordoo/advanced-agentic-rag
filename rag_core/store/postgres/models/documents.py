import datetime
import enum
from typing import List, Optional

from sqlalchemy import (
    String,
    Text,
    DateTime,
    UniqueConstraint,
    Index,
    Integer,
    ForeignKey,
    Enum as SQLEnum
)
from sqlalchemy.sql import func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
# Make sure you have pgvector installed: pip install pgvector
from pgvector.sqlalchemy import Vector

from rag_core.store.postgres.models.base import Base

VECTOR_DIMS=384

class IngestionStatus(str, enum.Enum):
    """State machine for the background ingestion worker."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentRegistry(Base):
    __tablename__ = 'document_registry'

    # --- Database Mechanics ---
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    file_id: Mapped[str] = mapped_column(String(255), unique=True, index=True)

    # --- Location Identity ---
    folder_name: Mapped[str] = mapped_column(String(255))
    file_name: Mapped[str] = mapped_column(String(255))
    source_uri: Mapped[str] = mapped_column(String(1024))

    # --- Content State ---
    content_hash: Mapped[str] = mapped_column(String(64))

    # --- State Machine & Metadata ---
    status: Mapped[IngestionStatus] = mapped_column(
        SQLEnum(IngestionStatus, name="ingestion_status_enum"),
        default=IngestionStatus.PENDING
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    ingestion_metadata: Mapped[Optional[dict]] = mapped_column(JSONB, server_default='{}')

    # --- Audit Trail ---
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        UniqueConstraint('folder_name', 'file_name', name='uix_folder_file'),
    )


class ParentDocument(Base):
    __tablename__ = 'parent_documents'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    registry_id: Mapped[int] = mapped_column(ForeignKey('document_registry.id', ondelete='CASCADE'))

    # Denormalized from registry for fast access without joins
    file_id: Mapped[str] = mapped_column(String(255), index=True)
    source_uri: Mapped[str] = mapped_column(String(1024))

    doc_type: Mapped[str] = mapped_column(String(50)) # e.g., 'text', 'table'
    # Using an array of integers to support spanning multiple pages
    page_numbers: Mapped[Optional[list[int]]] = mapped_column(ARRAY(Integer))
    text: Mapped[str] = mapped_column(Text)



class ChildDocument(Base):
    __tablename__ = 'child_documents'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    parent_id: Mapped[int] = mapped_column(ForeignKey('parent_documents.id', ondelete='CASCADE'))

    # Denormalized for ultra-fast hybrid search filtering
    file_id: Mapped[str] = mapped_column(String(255), index=True)
    page_numbers: Mapped[Optional[list[int]]] = mapped_column(ARRAY(Integer))

    # The actual semantic content
    text: Mapped[str] = mapped_column(Text)

    # Vector Column (AWe leave it empty and enforace it later via settigns)
    vector: Mapped[list[float]] = mapped_column(Vector())

    # Extensible filtering arrays (like Qdrant)
    scope: Mapped[Optional[list[str]]] = mapped_column(ARRAY(String), server_default='{}')
    tags: Mapped[Optional[list[str]]] = mapped_column(ARRAY(String), server_default='{}')

    # Catch-all for chunking strategy, character overlap, etc.
    chunk_metadata: Mapped[Optional[dict]] = mapped_column(JSONB, server_default='{}')

    # --- Relationships ---
    parent: Mapped["ParentDocument"] = relationship()

    __table_args__ = (
        # GIN Indexes to make the Array filtering as fast as Qdrant
        Index('ix_child_scope_gin', 'scope', postgresql_using='gin'),
        Index('ix_child_tags_gin', 'tags', postgresql_using='gin'),
    )
