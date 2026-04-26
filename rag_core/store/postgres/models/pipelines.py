from __future__ import annotations

import datetime
import enum
from typing import Optional

from sqlalchemy import DateTime, Enum as SQLEnum, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from rag_core.store.postgres.models.base import Base


class PipelineType(str, enum.Enum):
    """Pipeline families that can share run tracking."""

    INGESTION = "ingestion"
    EVALUATION = "evaluation"


class RunPhase(str, enum.Enum):
    """Recoverable phase marker for a pipeline run."""

    SCANNING = "scanning"
    RECONCILING = "reconciling"
    INGESTING = "ingesting"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    pipeline_type: Mapped[PipelineType] = mapped_column(
        SQLEnum(PipelineType, name="pipeline_type_enum",native_enum=False,length=50),
        default=PipelineType.INGESTION,
        nullable=False,
    )
    phase: Mapped[RunPhase] = mapped_column(
        SQLEnum(RunPhase, name="run_phase_enum",native_enum=False,length=50),
        default=RunPhase.SCANNING,
        nullable=False,
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text)

    started_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    completed_at: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime(timezone=True)
    )
