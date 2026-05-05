from __future__ import annotations

import datetime
import enum
from typing import Optional

from sqlalchemy import (
    ARRAY,
    DateTime,
    Enum as SQLEnum,
    ForeignKey,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from rag_core.store.postgres.models.base import Base


class IngestionRunState(str, enum.Enum):
    """Lifecycle state for an ingestion run."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class IngestionRunStep(str, enum.Enum):
    """Current processing phase for a running ingestion run."""

    SCANNING = "SCANNING"
    PARSING = "PARSING"
    CHUNKING = "CHUNKING"
    EMBEDDING = "EMBEDDING"
    INGESTING = "INGESTING"
    RECONCILING = "RECONCILING"


class IngestionRun(Base):
    __tablename__ = "ingestion_runs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    state: Mapped[IngestionRunState] = mapped_column(
        SQLEnum(
            IngestionRunState,
            name="run_state_enum",
            native_enum=False,
            validate_strings=True,
            create_constraint=True,
            length=50,
        ),
        default=IngestionRunState.PENDING,
        nullable=False,
    )
    step: Mapped[IngestionRunStep] = mapped_column(
        SQLEnum(
            IngestionRunStep,
            name="run_step_enum",
            native_enum=False,
            validate_strings=True,
            create_constraint=True,
            length=50,
        ),
        nullable=False,
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text)

    started_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    completed_at: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime(timezone=True)
    )
    metrics: Mapped[dict]= mapped_column(JSONB, server_default="{}", nullable=True)

    def __repr__(self) -> str:
        return (
            "IngestionRun("
            f"id={self.id}, "
            f"state={self.state}, "
            f"step={self.step}, "
            f"started_at={self.started_at}, "
            f"completed_at={self.completed_at}"
            ")"
        )


class AssetState(str, enum.Enum):
    """Latest processing state for a source asset."""

    PENDING = "PENDING"       # Ready to be chunked
    PROCESSING = "PROCESSING" # Worker currently has it
    COMPLETED = "COMPLETED"   # Successfully embedded
    FAILED = "FAILED"         # Worker crashed on this file
    DELETED = "DELETED"       # File vanished from the source drive
    EXCLUDED = "EXCLUDED"     # Human explicitly set "Do not read this"


class SourceAsset(Base):
    __tablename__ = "source_assets"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    asset_key: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    source_system: Mapped[str]= mapped_column(String(100),server_default="local_fs",nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64),nullable=False)
    state: Mapped[AssetState] = mapped_column(
        SQLEnum(
            AssetState,
            name="asset_state_enum",
            native_enum=False,
            validate_strings=True,
            create_constraint=True,
            length=50,
        ),
        default=AssetState.PENDING,
        nullable=False,
    )
    worker_id: Mapped[Optional[str]] = mapped_column(String(50))
    last_seen_run_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("ingestion_runs.id", ondelete="SET NULL"), index=True
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

    asset_name: Mapped[str] = mapped_column(Text,nullable=False)
    parent_name: Mapped[str] = mapped_column(Text,nullable=True)
    source_uri: Mapped[str] = mapped_column(Text,nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text)

    tags: Mapped[list[str]] = mapped_column(
            ARRAY(String), server_default="{}", nullable=False
        )
    scopes: Mapped[list[str]] = mapped_column(
        ARRAY(String), server_default="{}", nullable=False
    )
    asset_metadata: Mapped[dict] = mapped_column(
        JSONB, server_default="{}", nullable=False
    )





    last_seen_run: Mapped[Optional["IngestionRun"]] = relationship()

    __table_args__ = (
        UniqueConstraint(
            "parent_name", "asset_name", name="uix_asset_parent_name"
        ),
    )

    def __repr__(self) -> str:
        return (
            "SourceAsset("
            f"id={self.id}, "
            f"asset_key={self.asset_key!r}, "
            f"parent_name={self.parent_name!r}, "
            f"asset_name={self.asset_name!r}, "
            f"state={self.state}, "
            f"last_seen_run_id={self.last_seen_run_id}"
            ")"
        )
