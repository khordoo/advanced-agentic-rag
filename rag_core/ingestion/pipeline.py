"""Canonical ingestion pipeline orchestration entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
import logging

from rag_core.core.config import get_ingestion_settings, setup_logging
from rag_core.ingestion.indexing_preflight import prepare_indexing_preflight
from rag_core.ingestion.services.discovery import run_discovery
from rag_core.ingestion.services.discovery import DiscoveryRunMetrics
from rag_core.ingestion.services.worker import SourceIngestionWorkerService

setup_logging()
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EndToEndIngestionResult:
    """Result returned by the composed discovery + ingestion pipeline."""

    discovery: DiscoveryRunMetrics | None
    processed_jobs: int
    succeeded: bool


def run_discovery_pipeline() -> DiscoveryRunMetrics | None:
    """Run source discovery and queue assets for ingestion."""
    logger.info("Starting discovery pipeline")
    return run_discovery()


def run_ingestion_pipeline(
    *,
    overwrite: bool = False,
    max_jobs: int | None = None,
) -> int:
    """Run the ingestion worker against queued source assets."""
    preflight = prepare_indexing_preflight(overwrite=overwrite)
    logger.info(
        "Starting ingestion worker pipeline",
        extra={"overwrite": overwrite, "max_jobs": max_jobs},
    )
    worker = SourceIngestionWorkerService.default_production()
    effective_max_jobs = (
        max_jobs if max_jobs is not None else worker.settings.default_max_jobs
    )

    if overwrite:
        preflight.log_overwrite_start()
        worker.indexer.reset_collections()
        logger.info("Collections reset before ingestion pipeline")

    processed_jobs = worker.run(max_jobs=effective_max_jobs)
    if overwrite:
        preflight.log_overwrite_success()
    logger.info(
        "Ingestion worker pipeline completed",
        extra={"processed_jobs": processed_jobs},
    )
    return processed_jobs


def run_end_to_end_pipeline(
    *,
    overwrite: bool = False,
    max_jobs: int | None = None,
) -> EndToEndIngestionResult:
    """Run discovery followed by ingestion in a single canonical entrypoint."""
    prepare_indexing_preflight(overwrite=overwrite)
    logger.info(
        "Starting end-to-end ingestion pipeline",
        extra={"overwrite": overwrite, "max_jobs": max_jobs},
    )
    discovery = run_discovery_pipeline()

    if discovery is None:
        logger.error("Discovery failed; skipping ingestion worker pipeline")
        return EndToEndIngestionResult(
            discovery=None,
            processed_jobs=0,
            succeeded=False,
        )

    processed_jobs = run_ingestion_pipeline(
        overwrite=overwrite,
        max_jobs=max_jobs,
    )
    return EndToEndIngestionResult(
        discovery=discovery,
        processed_jobs=processed_jobs,
        succeeded=True,
    )


def run_pipeline() -> EndToEndIngestionResult:
    """Backward-compatible script entrypoint for the canonical e2e pipeline."""
    settings = get_ingestion_settings()
    return run_end_to_end_pipeline(overwrite=settings.overwrite)


if __name__ == "__main__":
    run_pipeline()
