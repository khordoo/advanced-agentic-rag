import datetime
import logging

from sqlalchemy.orm import Session
from sqlalchemy import update, func
from rag_core.store.postgres.models.ingestion import (
    IngestionRun,
    IngestionRunState,
    IngestionRunStep,
)

logger = logging.getLogger(__name__)


class IngestionRunRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create_run(self) -> IngestionRun:
        run = IngestionRun(
            state=IngestionRunState.RUNNING,
            step=IngestionRunStep.SCANNING,
        )
        self.session.add(run)
        self.session.flush()
        return run

    def mark_reconciling(self, run_id: int) -> IngestionRun:
        return self._update_run(run_id, step=IngestionRunStep.RECONCILING)

    def mark_ingesting(self, run_id: int) -> IngestionRun:
        return self._update_run(run_id, step=IngestionRunStep.INGESTING)

    def mark_completed(self, run_id: int, metrics:dict) -> IngestionRun:
        stmt = (
            update(IngestionRun)
            .where(IngestionRun.id == run_id)
            .values(
                state=IngestionRunState.COMPLETED,
                completed_at=datetime.datetime.now(datetime.UTC),
                metrics=metrics
            )
        )
        self.session.execute(stmt)

    def mark_failed(self, run_id: int, error_message: str) -> IngestionRun:
        stmt = (
            update(IngestionRun)
            .where(IngestionRun.id == run_id)
            .values(
                state=IngestionRunState.FAILED,
                completed_at=datetime.datetime.now(datetime.UTC),
            )
        )
        self.session.execute(stmt)

    def finalize_run(self, run_id: int, state: IngestionRunState, metrics: dict) -> None:
        stmt = (
            update(IngestionRun)
            .where(IngestionRun.id == run_id)
            .values(
                state=state,
                completed_at=func.now(),
                metrics=metrics
            )
        )
        self.session.execute(stmt)

    def _update_run(self, run_id: int, **kwargs) -> IngestionRun:
        run = self.session.get(IngestionRun, run_id)
        if run is None:
            raise RuntimeError(f"Ingestion run not found. ID: {run_id}")

        for key, value in kwargs.items():
            setattr(run, key, value)

        logger.info("Ingestion run updated: %s", run)
        return run
