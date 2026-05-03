from __future__ import annotations

import datetime
import enum
from typing import Optional

from sqlalchemy import DateTime, Enum as SQLEnum, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from rag_core.store.postgres.models.base import Base



class IngestionRunStatus(str, enum.Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

# 2. The Pipeline Steps (Where is it currently?)
class IngestionRunPhase(str, enum.Enum):
    SCANNING = "SCANNING"
    PARSING = "PARSING"
    CHUNKING = "CHUNKING"
    EMBEDDING = "EMBEDDING"
    INGESTING ="INGESTING"
    RECONCILING="RECONCILING"

class IngestionRun(Base):
    __tablename__ = "ingestion_runs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    status: Mapped[IngestionRunStatus] = mapped_column(
        SQLEnum(IngestionRunStatus, name="run_status_enum",native_enum=False,validate_strings=True,
                create_constraint=True, length=50),
        default=IngestionRunStatus.PENDING,
        nullable=False,
    )
    phase: Mapped[IngestionRunPhase] = mapped_column(
        SQLEnum(IngestionRunPhase, name="run_phase_enum",native_enum=False,validate_strings=True,
                create_constraint=True, length=50),
        nullable=False,
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text)

    started_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    completed_at: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime(timezone=True)
    )
