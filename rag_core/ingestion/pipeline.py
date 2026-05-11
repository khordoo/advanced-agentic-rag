"""Canonical ingestion pipeline orchestration entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
import logging

from rag_core.core.config import get_ingestion_settings, setup_logging
from rag_core.indexing.factory import (
    IndexingCompatibilityResult,
    validate_indexing_runtime_compatibility,
)
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


def _validate_runtime_indexing_config(
    *,
    overwrite: bool,
) -> IndexingCompatibilityResult:
    """Fail fast on incompatible indexer/embedding settings."""
    settings = get_ingestion_settings()
    return validate_indexing_runtime_compatibility(
        settings.indexer_backend,
        overwrite=overwrite,
    )


def _log_overwrite_start(preflight: IndexingCompatibilityResult) -> None:
    logger.warning(
        "INGESTION_OVERWRITE/overwrite is enabled. Existing vectors for this "
        "backend will be deleted."
    )
    if preflight.overwrite_rebuild:
        existing_dimensions = ", ".join(str(dim) for dim in preflight.existing_dimensions)
        logger.warning(
            "Dimension mismatch detected (DB: %s, Config: %s). Proceeding with "
            "index reset as requested.",
            existing_dimensions,
            preflight.configured_dimension,
        )


def _log_overwrite_success(preflight: IndexingCompatibilityResult) -> None:
    if not preflight.overwrite_rebuild:
        return
    logger.info(
        "Overwrite successful. Index rebuilt with dimension %s.",
        preflight.configured_dimension,
    )
    logger.warning(
        "Do not keep INGESTION_OVERWRITE=true in long-lived config; run future "
        "ingestion with overwrite=False to prevent accidental data loss."
    )


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
    preflight = _validate_runtime_indexing_config(overwrite=overwrite)
    logger.info(
        "Starting ingestion worker pipeline",
        extra={"overwrite": overwrite, "max_jobs": max_jobs},
    )
    worker = SourceIngestionWorkerService.default_production()
    effective_max_jobs = (
        max_jobs if max_jobs is not None else worker.settings.default_max_jobs
    )

    if overwrite:
        _log_overwrite_start(preflight)
        worker.indexer.reset_collections()
        logger.info("Collections reset before ingestion pipeline")

    processed_jobs = worker.run(max_jobs=effective_max_jobs)
    if overwrite:
        _log_overwrite_success(preflight)
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
    _validate_runtime_indexing_config(overwrite=overwrite)
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
