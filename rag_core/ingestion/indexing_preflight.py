"""Ingestion-facing indexing preflight and overwrite logging."""

from __future__ import annotations

from dataclasses import dataclass
import logging

from rag_core.core.config import get_ingestion_settings
from rag_core.indexing.factory import (
    IndexingCompatibilityResult,
    validate_indexing_runtime_compatibility,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionIndexingPreflight:
    """Prepared indexing preflight state for one ingestion run."""

    compatibility: IndexingCompatibilityResult

    def log_overwrite_start(self) -> None:
        """Log destructive overwrite warnings before the index reset happens."""
        logger.warning(
            "INGESTION_OVERWRITE/overwrite is enabled. Existing vectors for this "
            "backend will be deleted."
        )
        if self.compatibility.overwrite_rebuild:
            existing_dimensions = ", ".join(
                str(dim) for dim in self.compatibility.existing_dimensions
            )
            logger.warning(
                "Dimension mismatch detected (DB: %s, Config: %s). Proceeding with "
                "index reset as requested.",
                existing_dimensions,
                self.compatibility.configured_dimension,
            )

    def log_overwrite_success(self) -> None:
        """Log confirmation after a mismatch-triggered overwrite rebuild succeeds."""
        if not self.compatibility.overwrite_rebuild:
            return
        logger.info(
            "Overwrite successful. Index rebuilt with dimension %s.",
            self.compatibility.configured_dimension,
        )
        logger.warning(
            "Do not keep INGESTION_OVERWRITE=true in long-lived config; run future "
            "ingestion with overwrite=False to prevent accidental data loss."
        )


def prepare_indexing_preflight(*, overwrite: bool) -> IngestionIndexingPreflight:
    """Fail fast on incompatible indexer/embedding settings."""
    settings = get_ingestion_settings()
    compatibility = validate_indexing_runtime_compatibility(
        settings.indexer_backend,
        overwrite=overwrite,
    )
    return IngestionIndexingPreflight(compatibility=compatibility)
