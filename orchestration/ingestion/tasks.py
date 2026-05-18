"""Prefect tasks that wrap core ingestion entrypoints."""

from __future__ import annotations

import logging
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import text

from prefect.exceptions import MissingContextError
from prefect import get_run_logger
from prefect import task

from rag_core.ingestion.pipeline import run_discovery_pipeline
from rag_core.ingestion.pipeline import run_end_to_end_pipeline
from rag_core.ingestion.pipeline import run_ingestion_pipeline
from rag_core.ingestion.services.worker import SourceIngestionWorkerService
from rag_core.store.postgres.session import get_database


@task(name="discovery-task", retries=1, retry_delay_seconds=5)
def discovery_task() -> dict[str, int] | None:
    """Run discovery and return serializable metrics."""
    logger = _get_logger()
    logger.info("Starting discovery task")
    metrics = run_discovery_pipeline()
    if metrics is None:
        logger.error("Discovery returned no metrics (run failed)")
        return None
    logger.info("Discovery task completed")
    return metrics.model_dump()


@task(name="ingestion-task", retries=1, retry_delay_seconds=5)
def ingestion_task(*, max_jobs: int | None = None) -> int:
    """Run worker ingestion loop and return processed job count."""
    logger = _get_logger()
    logger.info("Starting ingestion task")
    try:
        processed_jobs = run_ingestion_pipeline(max_jobs=max_jobs)
    except Exception as exc:
        _raise_with_migration_guidance(exc)
    logger.info("Ingestion task completed")
    return processed_jobs


@task(name="end-to-end-ingestion-task", retries=1, retry_delay_seconds=5)
def end_to_end_ingestion_task(
    *,
    max_jobs: int | None = None,
) -> dict[str, object]:
    """Run discovery then ingestion and return a serializable result payload."""
    logger = _get_logger()
    logger.info("Starting end-to-end ingestion task")
    try:
        result = run_end_to_end_pipeline(max_jobs=max_jobs)
    except Exception as exc:
        _raise_with_migration_guidance(exc)

    payload = {
        "succeeded": result.succeeded,
        "processed_jobs": result.processed_jobs,
        "discovery": result.discovery.model_dump() if result.discovery else None,
    }
    if not result.succeeded:
        msg = (
            "End-to-end ingestion flow reported failure "
            "(discovery or ingestion phase unsuccessful)."
        )
        logger.error(msg)
        raise RuntimeError(msg)
    logger.info("End-to-end ingestion task completed")
    return payload


@task(name="run-database-migrations-task")
def run_database_migrations_task() -> None:
    """Apply Alembic migrations to head before ingestion work begins.

    This is an intentional **fail-fast preflight gate**: it runs as the first
    task inside ``end_to_end_flow`` so that the relational application schema
    is verified and updated *dynamically at runtime*, before any document
    discovery or text chunking takes place.  If the migration fails, the
    entire flow aborts immediately — preventing silent table-mismatch errors
    downstream.
    """
    logger = _get_logger()
    logger.info(
        "Running database migrations to head (fail-fast preflight gate)"
    )
    alembic_config = Config(str(_project_root() / "db" / "alembic.ini"))
    try:
        command.upgrade(alembic_config, "head")
    except Exception:
        logger.exception(
            "Database migration preflight FAILED — aborting flow to prevent "
            "silent schema drift"
        )
        raise
    logger.info("Database migrations completed — schema is current ✓")


@task(name="reset-vector-backend-task")
def reset_vector_backend_task() -> None:
    """Destructively clear the configured vector backend."""
    logger = _get_logger()
    logger.warning("Resetting vector backend collections")
    worker = SourceIngestionWorkerService.default_production()
    worker.indexer.reset_collections()
    logger.warning("Vector backend collections reset completed")


@task(name="reset-ingestion-registries-task")
def reset_ingestion_registries_task() -> None:
    """Destructively clear ingestion state tables."""
    logger = _get_logger()
    logger.warning("Resetting ingestion registries")
    database = get_database()
    with database.get_session() as session:
        session.execute(
            text("TRUNCATE TABLE source_assets, ingestion_runs RESTART IDENTITY CASCADE")
        )
        session.commit()
    logger.warning("Ingestion registries reset completed")


def _raise_with_migration_guidance(exc: Exception) -> None:
    lowered = str(exc).lower()
    migration_signals = (
        "undefinedtable",
        "no such table",
        "relation",
        "does not exist",
        "alembic_version",
    )
    if any(signal in lowered for signal in migration_signals):
        raise RuntimeError(
            "Database schema appears out of date. Run migrations with `make migrate-up` "
            "and retry the ingestion flow."
        ) from exc
    raise


def _get_logger() -> logging.Logger:
    try:
        return get_run_logger()
    except MissingContextError:
        return logging.getLogger(__name__)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]
