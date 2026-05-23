"""Lease-aware source ingestion worker service."""

from __future__ import annotations

import logging
from pathlib import Path
import random
import socket
import time
import uuid

from rag_core.core.config import IngestionSettings, get_ingestion_settings
from rag_core.indexing.base import BaseIndexer
from rag_core.ingestion.processing.base import BaseProcessor
from rag_core.schemas import ClaimedSourceAsset, ParentChunk
from rag_core.store.postgres.models.ingestion import AssetState
from rag_core.store.postgres.repositories.source_assets import SourceAssetRepository
from rag_core.store.postgres.session import Database, get_database

logger = logging.getLogger(__name__)


class SimulatedAssetProcessor(BaseProcessor):
    """Development processor that simulates work without producing chunks."""

    def __init__(
        self,
        min_delay_seconds: float = 1.0,
        max_delay_seconds: float = 3.0,
    ) -> None:
        if min_delay_seconds < 0:
            raise ValueError("min_delay_seconds must be greater than or equal to zero")
        if max_delay_seconds < min_delay_seconds:
            raise ValueError(
                "max_delay_seconds must be greater than or equal to min_delay_seconds"
            )

        self.min_delay_seconds = min_delay_seconds
        self.max_delay_seconds = max_delay_seconds

    def process_asset(self, asset: ClaimedSourceAsset) -> list[ParentChunk]:
        delay_seconds = random.uniform(
            self.min_delay_seconds,
            self.max_delay_seconds,
        )
        logger.info(
            "Simulating asset processing",
            extra={
                "asset_id": asset.id,
                "source_uri": asset.source_uri,
                "delay_seconds": delay_seconds,
            },
        )
        time.sleep(delay_seconds)
        logger.info(
            "Finished simulated asset processing",
            extra={"asset_id": asset.id, "source_uri": asset.source_uri},
        )
        return []


class LoggingIndexer(BaseIndexer):
    """Development indexer that logs received chunks without writing them."""

    def index_asset(
        self,
        asset: ClaimedSourceAsset,
        document_blocks: list[ParentChunk],
    ) -> None:
        logger.info(
            "Simulated indexer received document blocks",
            extra={
                "asset_id": asset.id,
                "asset_key": asset.asset_key,
                "parent_chunks": len(document_blocks),
            },
        )

    def reset_collections(self) -> None:
        logger.info("Simulated indexer reset requested")


class SourceIngestionWorkerService:
    """Claim, process, index, and finalize source asset ingestion jobs."""

    def __init__(
        self,
        db: Database | None = None,
        settings: IngestionSettings | None = None,
        processor: BaseProcessor | None = None,
        indexer: BaseIndexer | None = None,
        worker_id: str | None = None,
    ) -> None:
        if processor is None or indexer is None:
            raise ValueError(
                "SourceIngestionWorkerService requires both processor and indexer. "
                "Use SourceIngestionWorkerService.simulated(...) for local simulation."
            )

        self.db = db or get_database()
        self.settings = settings or get_ingestion_settings()
        self.processor = processor
        self.indexer = indexer
        self.worker_id = worker_id or self._build_worker_id()

    @classmethod
    def simulated(
        cls,
        db: Database | None = None,
        settings: IngestionSettings | None = None,
        *,
        min_delay_seconds: float = 1.0,
        max_delay_seconds: float = 3.0,
        worker_id: str | None = None,
    ) -> SourceIngestionWorkerService:
        """Build a worker that exercises the lease flow without real indexing."""
        return cls(
            db=db,
            settings=settings,
            processor=SimulatedAssetProcessor(
                min_delay_seconds=min_delay_seconds,
                max_delay_seconds=max_delay_seconds,
            ),
            indexer=LoggingIndexer(),
            worker_id=worker_id,
        )

    @classmethod
    def default_production(
        cls,
        db: Database | None = None,
        settings: IngestionSettings | None = None,
        worker_id: str | None = None,
    ) -> SourceIngestionWorkerService:
        """Builds a production-ready worker using the configured defaults.

        The indexer backend (Qdrant or Postgres) is resolved from
        ``IngestionSettings.indexer_backend`` via the factory.
        """
        from rag_core.indexing.factory import create_indexer
        from rag_core.ingestion.processing.markdown_processor import MarkdownProcessor

        resolved_settings = settings or get_ingestion_settings()

        return cls(
            db=db,
            settings=resolved_settings,
            processor=MarkdownProcessor(),
            indexer=create_indexer(resolved_settings.indexer_backend),
            worker_id=worker_id,
        )

    def run(self, max_jobs: int | None = None) -> int:
        """Process jobs until the queue is empty or max_jobs is reached."""
        if max_jobs is not None and max_jobs < 0:
            raise ValueError("max_jobs must be greater than or equal to zero")

        processed_jobs = 0
        logger.info(
            "Worker starting ingestion loop",
            extra={"worker_id": self.worker_id, "max_jobs": max_jobs},
        )

        while max_jobs is None or processed_jobs < max_jobs:
            if not self.run_once():
                break
            processed_jobs += 1

        logger.info(
            "Worker finished ingestion loop",
            extra={"worker_id": self.worker_id, "processed_jobs": processed_jobs},
        )
        return processed_jobs

    def run_once(self) -> bool:
        """Claim and process one source asset if work is available."""
        asset = self._claim_next_asset()

        if asset is None:
            logger.info(
                "No claimable source assets found",
                extra={"worker_id": self.worker_id},
            )
            return False

        try:
            logger.info(
                "Worker claimed source asset",
                extra={"worker_id": self.worker_id, "asset_id": asset.id},
            )

            chunks = self._process_asset(asset)

            if not self._extend_asset_lease(
                asset.id,
                lease_version=asset.lease_version,
                extra_seconds=self.settings.worker_preflight_extension_seconds,
            ):
                logger.warning(
                    "Lost source asset lease before indexing; skipping index write",
                    extra={"worker_id": self.worker_id, "asset_id": asset.id},
                )
                return True

            self._index_chunks(asset, chunks)

            if self._mark_asset_complete(asset.id, lease_version=asset.lease_version):
                logger.info(
                    "Source asset completed",
                    extra={"worker_id": self.worker_id, "asset_id": asset.id},
                )
            else:
                logger.critical(
                    "Indexed source asset could not be marked complete; "
                    "deterministic chunk IDs are required for safe retries",
                    extra={"worker_id": self.worker_id, "asset_id": asset.id},
                )
        except Exception as exc:
            error_payload = self._build_asset_error_message(asset=asset, exc=exc)
            logger.exception(
                "Failure processing source asset",
                extra={
                    "worker_id": self.worker_id,
                    "asset_id": asset.id,
                    "source_uri": asset.source_uri,
                    "error_payload": error_payload,
                },
            )
            new_state = self._mark_asset_failed(
                asset.id,
                lease_version=asset.lease_version,
                error_message=error_payload,
            )
            if new_state == AssetState.PENDING:
                logger.warning(
                    "Asset processing failed; returning asset to queue",
                    extra={
                        "worker_id": self.worker_id,
                        "asset_id": asset.id,
                        "asset_state": new_state.value,
                    },
                )
            elif new_state == AssetState.FAILED:
                logger.critical(
                    "Asset exhausted retries; permanently marking failed",
                    extra={
                        "worker_id": self.worker_id,
                        "asset_id": asset.id,
                        "asset_state": new_state.value,
                    },
                )
            else:
                logger.warning(
                    "Could not mark failed source asset; worker no longer owns lease",
                    extra={"worker_id": self.worker_id, "asset_id": asset.id},
                )

        return True

    @staticmethod
    def _build_worker_id() -> str:
        hostname = socket.gethostname()
        short_uuid = uuid.uuid4().hex[:8]
        return f"worker-{hostname}-{short_uuid}"

    def _claim_next_asset(self) -> ClaimedSourceAsset | None:
        with self.db.get_session() as session:
            with session.begin():
                repo = SourceAssetRepository(session)
                asset = repo.claim_next_asset(
                    worker_id=self.worker_id,
                    max_attempts=self.settings.worker_max_attempts,
                    lease_duration_seconds=self.settings.worker_lease_duration_seconds,
                )

        logger.debug(
            "Source asset claim attempt finished",
            extra={
                "worker_id": self.worker_id,
                "asset_id": asset.id if asset else None,
            },
        )
        return asset

    def _mark_asset_complete(self, asset_id: int, *, lease_version: int) -> bool:
        with self.db.get_session() as session:
            with session.begin():
                repo = SourceAssetRepository(session)
                return repo.mark_asset_complete_if_owned(
                    expected_version=lease_version,
                    asset_id=asset_id,
                )

    def _extend_asset_lease(
        self, asset_id: int, *, lease_version: int, extra_seconds: int
    ) -> bool:
        with self.db.get_session() as session:
            with session.begin():
                repo = SourceAssetRepository(session)
                return repo.extend_asset_lease(
                    expected_version=lease_version,
                    asset_id=asset_id,
                    extra_seconds=extra_seconds,
                )

    def _mark_asset_failed(
        self,
        asset_id: int,
        *,
        lease_version: int,
        error_message: str,
    ) -> AssetState | None:
        with self.db.get_session() as session:
            with session.begin():
                repo = SourceAssetRepository(session)
                return repo.mark_asset_failed_or_retry_if_owned(
                    expected_version=lease_version,
                    asset_id=asset_id,
                    error_message=error_message,
                    max_attempts=self.settings.worker_max_attempts,
                )

    def _process_asset(self, asset: ClaimedSourceAsset) -> list[ParentChunk]:
        return self.processor.process_asset(asset)

    @staticmethod
    def _build_asset_error_message(asset: ClaimedSourceAsset, exc: Exception) -> str:
        source_uri = asset.source_uri
        resolved_path = source_uri
        file_ext = Path(source_uri).suffix.lower()
        stage = "asset_processing"

        if "file://" in source_uri:
            resolved_path = source_uri.replace("file://", "", 1)

        message = str(exc)
        lowered = message.lower()
        if "convert_pdf_to_markdown" in lowered or "pdf ingestion requires" in lowered:
            stage = "convert_pdf_to_markdown"
        elif "table" in lowered and "summar" in lowered:
            stage = "table_summarization"
        elif "chunk" in lowered:
            stage = "chunk_markdown"

        return (
            f"type={type(exc).__name__}; stage={stage}; asset_id={asset.id}; "
            f"asset_key={asset.asset_key}; source_uri={source_uri}; "
            f"resolved_path={resolved_path}; ext={file_ext}; detail={message}"
        )

    def _index_chunks(
        self,
        asset: ClaimedSourceAsset,
        chunks: list[ParentChunk],
    ) -> None:
        self.indexer.index_asset(asset, chunks)

if __name__ == "__main__":
    from rag_core.core.config import setup_logging
    setup_logging()
    SourceIngestionWorkerService.simulated().run()
