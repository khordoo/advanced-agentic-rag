"""Legacy ingestion pipeline — DEPRECATED.

.. deprecated::
    This module is a backward-compatibility shim.  All new code should use
    :class:`~rag_core.ingestion.services.worker.SourceIngestionWorkerService`
    directly.  This shim will be removed in a future release.

The ``IngestionPipeline`` class now delegates to
:meth:`SourceIngestionWorkerService.default_production().run()`.
"""

from __future__ import annotations

import logging
import warnings

from rag_core.core.config import load_config, setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Deprecated wrapper around the modern worker-based ingestion service.

    .. deprecated::
        Use ``SourceIngestionWorkerService.default_production()`` instead.
    """

    def __init__(self, config: dict | None = None, overwrite: bool = True, **kwargs):
        warnings.warn(
            "IngestionPipeline is deprecated and will be removed in a future release. "
            "Use SourceIngestionWorkerService.default_production() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.config = config or {}
        self.overwrite = overwrite
        logger.info(
            "IngestionPipeline (deprecated shim) initialised. "
            "Delegates to SourceIngestionWorkerService."
        )

    def run(self):
        """Execute ingestion via the modern worker service."""
        from rag_core.ingestion.services.worker import SourceIngestionWorkerService

        logger.info("IngestionPipeline.run() delegating to worker service …")
        worker = SourceIngestionWorkerService.default_production()

        if self.overwrite:
            worker.indexer.reset_collections()
            logger.info("Collections reset (overwrite=True)")

        processed = worker.run()
        logger.info("Worker completed: %d jobs processed", processed)
        return processed


if __name__ == "__main__":
    cfg = load_config()
    p = IngestionPipeline(config=cfg, overwrite=True)
    p.run()
