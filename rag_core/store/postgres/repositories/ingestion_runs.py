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
        """Initializes a new run. started_at is handled by DB server_default."""
        run = IngestionRun(
            state=IngestionRunState.RUNNING,
            step=IngestionRunStep.SCANNING,
        )
        self.session.add(run)
        self.session.flush()
        return run

    # --- PROGRESS METHODS ---

    def update_progress(
        self,
        run_id: int,
        step: IngestionRunStep,
        state: IngestionRunState = IngestionRunState.RUNNING,
        metrics: dict | None = None
    ) -> None:
        """Updates milestones (scanning_at, reconciling_at, ingesting_at)."""
        milestone_map = {
            IngestionRunStep.SCANNING: "scanning_at",
            IngestionRunStep.RECONCILING: "reconciling_at",
            IngestionRunStep.INGESTING: "ingesting_at",
        }

        update_data = {"step": step, "state": state}
        if metrics:
            update_data["metrics"] = metrics

        col_name = milestone_map.get(step)
        if col_name:
            update_data[col_name] = func.now()

        self._update_run(run_id, **update_data)

    # --- FINALIZATION METHODS ---

    def complete(self, run_id: int, metrics: dict) -> None:
        """Closes the run successfully."""
        self._update_run(
            run_id,
            state=IngestionRunState.COMPLETED,
            completed_at=func.now(),
            metrics=metrics
        )

    def fail(self, run_id: int, error: str | None = None) -> None:
        """Closes the run with a failure state."""
        update_data = {
            "state": IngestionRunState.FAILED,
            "completed_at": func.now(),
        }
        if error:
           update_data["error_message"]=error
        # If you add an 'error_log' column later, add it here
        self._update_run(run_id, **update_data)

    # --- THE ENGINE ---

    def _update_run(self, run_id: int, **values) -> None:
        stmt = (
            update(IngestionRun)
            .where(IngestionRun.id == run_id)
            .values(**values)
        )
        self.session.execute(stmt)
        logger.info("Run %s status update: %s", run_id, values)
