from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
import logging
from typing import Any

from pydantic import BaseModel, Field

from rag_core.core.config import IngestionSettings, get_ingestion_settings
from rag_core.ingestion.eligibility import (
    EligibilityDecision,
    EligibilityResult,
    IngestionEligibilityService,
)
from rag_core.ingestion.metadata import extract_metadata_from_uri
from rag_core.ingestion.scanner import FileScanner, SourceAssetScanner
from rag_core.store.postgres.models.ingestion import (
    AssetState,
    IngestionRunState,
    IngestionRunStep,
    SourceAsset,
)
from rag_core.store.postgres.repositories.ingestion_runs import IngestionRunRepository
from rag_core.store.postgres.repositories.source_assets import SourceAssetRepository
from rag_core.store.postgres.session import Database, get_database
from rag_core.store.storage.provider import LocalFileSourceProvider, SourceProvider

logger = logging.getLogger(__name__)


class DiscoveryRunMetrics(BaseModel):
    """Strongly typed state tracker for pipeline execution."""

    discovered_total: int = Field(default=0)
    queued_new: int = Field(default=0)
    queued_reprocess: int = Field(default=0)
    skipped_unchanged: int = Field(default=0)
    failed: int = Field(default=0)
    deleted: int = Field(default=0)


class SourceDiscoveryService:
    """Coordinate the discovery phase of an ingestion run.

    This service scans the configured source location, evaluates each discovered
    asset for ingestion eligibility, queues new or changed assets for downstream
    processing, and reconciles previously known assets that are no longer present
    in the source. It owns the ingestion-run lifecycle transitions for discovery:
    creating a run, marking scanning and reconciliation progress, and then handing
    the run off to the ingestion worker phase.

    Dependencies are injected so discovery remains storage- and database-aware
    without hard-coding either concern. Per-asset failures are isolated using
    nested transactions, ensuring one corrupted file does not abort the entire
    scan, while catastrophic failures gracefully mark the ingestion run as failed
    where possible.
    """

    def __init__(
        self,
        settings: IngestionSettings,
        scanner: SourceAssetScanner,
        source_provider: SourceProvider,
        db: Database,
    ) -> None:
        self.settings = settings
        self.source_provider = source_provider
        self.scanner = scanner
        self.db = db
        self.run_id: int | None = None

    def run(self) -> DiscoveryRunMetrics | None:
        """Executes the discovery pipeline and manages the ingestion run state.

        This method orchestrates the full discovery lifecycle: initialization,
        scanning, reconciliation, and worker handoff. It serves as the primary
        error boundary for the service, catching unhandled exceptions to ensure
        the run is marked as failed in the database before exiting.

        Returns:
            DiscoveryRunMetrics: The final aggregated metrics if the run succeeds.
            None: If a catastrophic failure occurs and the run is aborted.
        """

        try:
            self._initialize()
            metrics = self._execute_scan_loop()
            metrics.deleted += self._reconcile_source_assets()
            self._transition_to_ingestion(metrics)
            return metrics

        except Exception as exc:
            logger.exception("Catastrophic failure in discovery run run_id=%s", self.run_id)
            try:
                self._handle_failure(str(exc))
            except Exception:
                logger.exception("Failed to mark discovery run failed run_id=%s", self.run_id)
            return None

    def _initialize(self) -> None:
        with self.db.get_session() as session:
            with session.begin():
                repo = IngestionRunRepository(session)
                run = repo.create_run()
                self.run_id = run.id
                logger.info("Created ingestion run run_id=%s", run.id)

    def _execute_scan_loop(self) -> DiscoveryRunMetrics:
        """The core work loop. Raises exceptions if something catastrophic happens."""
        self._update_progress(step=IngestionRunStep.SCANNING)
        metrics = DiscoveryRunMetrics()

        with self.db.get_session() as session:
            asset_repo = SourceAssetRepository(session)
            eligibility_service = IngestionEligibilityService(
                asset_repo,
                self.source_provider,
                hash_algorithm=self.settings.hash_algorithm,
            )

            asset_uris = self.scanner.discover_assets(self.settings.base_source_uri)
            for batch in self._iter_batches(asset_uris, self.settings.discovery_batch_size):
                with session.begin():
                    for asset_uri in batch:
                        self._process_asset_with_savepoint(
                            asset_uri=asset_uri,
                            asset_repo=asset_repo,
                            eligibility_service=eligibility_service,
                            metrics=metrics,
                            session=session,
                        )

        logger.info("Scan loop completed run_id=%s metrics=%s", self.run_id, metrics.model_dump())
        return metrics

    def _process_asset_with_savepoint(
        self,
        asset_uri: str,
        asset_repo: SourceAssetRepository,
        eligibility_service: IngestionEligibilityService,
        metrics: DiscoveryRunMetrics,
        session: Any,
    ) -> None:
        metrics.discovered_total += 1
        current_asset_pk = None

        try:
            with session.begin_nested():
                logger.debug("Evaluating asset_uri=%s", asset_uri)
                eval_result = eligibility_service.evaluate_file(asset_uri)
                current_asset_pk = eval_result.existing_asset_pk

                self._apply_eligibility_decision(
                    asset_uri=asset_uri,
                    eval_result=eval_result,
                    asset_repo=asset_repo,
                    metrics=metrics,
                )
                session.flush()

        except Exception:
            metrics.failed += 1
            logger.exception("Discovery failed for asset_uri=%s", asset_uri)
            if current_asset_pk is not None:
                self._mark_asset_failed_with_savepoint(
                    asset_repo=asset_repo,
                    session=session,
                    asset_id=current_asset_pk,
                )

    def _apply_eligibility_decision(
        self,
        asset_uri: str,
        eval_result: EligibilityResult,
        asset_repo: SourceAssetRepository,
        metrics: DiscoveryRunMetrics,
    ) -> None:
        logger.debug(
            "Eligibility evaluated for asset_uri=%s decision=%s should_process=%s existing_asset_pk=%s",
            asset_uri,
            eval_result.decision.name,
            eval_result.should_process,
            eval_result.existing_asset_pk,
        )

        if not eval_result.should_process:
            self._mark_asset_seen(asset_repo, eval_result.existing_asset_pk)
            logger.debug("Skipped asset_uri=%s decision=%s", asset_uri, eval_result.decision.name)
            metrics.skipped_unchanged += 1
            return

        if eval_result.decision == EligibilityDecision.PROCESS_NEW:
            self._queue_new_asset(asset_uri, eval_result, asset_repo)
            metrics.queued_new += 1
            return

        if eval_result.decision == EligibilityDecision.REPROCESS_UPDATED:
            self._queue_updated_asset(asset_uri, eval_result, asset_repo)
            metrics.queued_reprocess += 1
            return

        logger.warning(
            "Unhandled eligibility decision asset_uri=%s decision=%s",
            asset_uri,
            eval_result.decision.name,
        )

    def _queue_new_asset(
        self,
        asset_uri: str,
        eval_result: EligibilityResult,
        asset_repo: SourceAssetRepository,
    ) -> None:
        metadata = extract_metadata_from_uri(asset_uri)
        asset = SourceAsset(
            asset_key=self._require_value(eval_result.asset_key, "asset_key"),
            content_hash=self._require_value(eval_result.content_hash, "content_hash"),
            source_uri=self.source_provider.get_access_url(asset_uri),
            asset_name=metadata.file_name,
            parent_name=metadata.folder_name,
            tags=metadata.tags,
            scopes=metadata.scopes,
            asset_metadata=metadata.extended_metadata,
            source_system=self.source_provider.source_system,
            last_seen_run_id=self._require_run_id(),
            state=AssetState.PENDING,
        )
        asset_repo.create_asset(asset)
        logger.info(
            "Queued new asset for processing asset_uri=%s asset_key=%s",
            asset_uri,
            eval_result.asset_key,
        )

    def _queue_updated_asset(
        self,
        asset_uri: str,
        eval_result: EligibilityResult,
        asset_repo: SourceAssetRepository,
    ) -> None:
        asset_repo.update_asset(
            asset_id=self._require_value(eval_result.existing_asset_pk, "existing_asset_pk"),
            content_hash=self._require_value(eval_result.content_hash, "content_hash"),
            state=AssetState.PENDING,
            last_seen_run_id=self._require_run_id(),
            updated_at=datetime.now(UTC),
        )
        logger.info(
            "Queued existing asset for reprocessing asset_uri=%s asset_pk=%s",
            asset_uri,
            eval_result.existing_asset_pk,
        )

    def _mark_asset_seen(
        self,
        asset_repo: SourceAssetRepository,
        asset_id: int | None,
    ) -> None:
        if asset_id is None:
            return

        asset_repo.update_asset(
            asset_id,
            last_seen_run_id=self._require_run_id(),
        )

    def _mark_asset_failed_with_savepoint(
        self,
        asset_repo: SourceAssetRepository,
        session: Any,
        asset_id: int,
    ) -> None:
        try:
            with session.begin_nested():
                asset_repo.update_asset(
                    asset_id=asset_id,
                    state=AssetState.FAILED,
                    last_seen_run_id=self._require_run_id(),
                )
                session.flush()
        except Exception:
            logger.exception("Failed to update asset state to FAILED asset_id=%s", asset_id)

    @staticmethod
    def _iter_batches(asset_uris: Iterable[str], batch_size: int) -> Iterator[list[str]]:
        batch: list[str] = []

        for asset_uri in asset_uris:
            batch.append(asset_uri)
            if len(batch) >= batch_size:
                yield batch
                batch = []

        if batch:
            yield batch

    def _reconcile_source_assets(self) -> int:
        """Handles the transactional logic for soft-deleting absent assets."""
        logger.info("Starting reconciliation for system: %s", self.source_provider.source_system)
        self._update_progress(step=IngestionRunStep.RECONCILING)

        with self.db.get_session() as session:
            with session.begin():
                repo = SourceAssetRepository(session)
                affected_count = repo.mark_deleted_absent_assets(
                    source_system=self.source_provider.source_system,
                    current_run_id=self._require_run_id(),
                )

        logger.info("Reconciliation completed run_id=%s, deleted=%d", self.run_id, affected_count)
        return affected_count

    def _update_progress(
        self,
        step: IngestionRunStep,
        state: IngestionRunState = IngestionRunState.RUNNING,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        with self.db.get_session() as session:
            with session.begin():
                runs_repo = IngestionRunRepository(session)
                runs_repo.update_progress(
                    self._require_run_id(),
                    step=step,
                    state=state,
                    metrics=metrics,
                )
                logger.info(
                    "Progress set run_id=%s step=%s state=%s",
                    self.run_id,
                    step,
                    state,
                )

    def _transition_to_ingestion(self, metrics: DiscoveryRunMetrics) -> None:
        """Signals that Discovery is done and the Worker can take over."""
        self._update_progress(
            step=IngestionRunStep.INGESTING,
            state=IngestionRunState.RUNNING,
            metrics=metrics.model_dump(),
        )
        logger.info("Discovery phase complete. Run %s now awaiting worker(s).", self.run_id)

    def _handle_failure(self, error: str) -> None:
        if self.run_id is None:
            logger.error("Discovery failed before an ingestion run was created: %s", error)
            return

        with self.db.get_session() as session:
            with session.begin():
                repo = IngestionRunRepository(session)
                repo.fail(self.run_id, error)
                logger.info("Marked ingestion run failed run_id=%s", self.run_id)

    def _require_run_id(self) -> int:
        if self.run_id is None:
            raise RuntimeError("Discovery run has not been initialized")
        return self.run_id

    @staticmethod
    def _require_value(value: Any | None, name: str) -> Any:
        if value is None:
            raise ValueError(f"Eligibility result is missing {name}")
        return value


def run_discovery() -> DiscoveryRunMetrics | None:
    settings = get_ingestion_settings()
    provider = LocalFileSourceProvider()
    scanner = FileScanner(provider, settings)
    db = get_database()

    service = SourceDiscoveryService(
        settings=settings,
        scanner=scanner,
        source_provider=provider,
        db=db,
    )
    return service.run()


if __name__ == "__main__":
    run_discovery()
