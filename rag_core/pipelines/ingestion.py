"""Legacy ingestion pipeline — DEPRECATED.

.. deprecated::
    This module is a backward-compatibility shim.  All new code should use
    ``rag_core.ingestion.pipeline``.  This shim will be removed in a future
    release.

The ``IngestionPipeline`` class now delegates to the canonical
``rag_core.ingestion.pipeline`` entrypoints.
"""

from __future__ import annotations

import logging
import warnings

from rag_core.core.config import load_config, setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Deprecated wrapper around the canonical ingestion pipeline.

    .. deprecated::
        Use ``rag_core.ingestion.pipeline.run_ingestion_pipeline`` instead.
    """

    def __init__(self, config: dict | None = None, overwrite: bool = True, **kwargs):
        warnings.warn(
            "IngestionPipeline is deprecated and will be removed in a future release. "
            "Use rag_core.ingestion.pipeline.run_ingestion_pipeline() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.config = config or {}
        self.overwrite = overwrite
        logger.info(
            "IngestionPipeline (deprecated shim) initialised. "
            "Delegates to rag_core.ingestion.pipeline."
        )

    def run(self):
        """Execute ingestion via the canonical pipeline entrypoint."""
        from rag_core.ingestion.pipeline import run_ingestion_pipeline

        logger.info("IngestionPipeline.run() delegating to canonical pipeline ...")
        processed = run_ingestion_pipeline(overwrite=self.overwrite)
        logger.info("Worker completed: %d jobs processed", processed)
        return processed


if __name__ == "__main__":
    cfg = load_config()
    p = IngestionPipeline(config=cfg, overwrite=True)
    p.run()
