import logging
import datetime
from pydantic import BaseModel ,Field
from rag_core.core.config import IngestionSettings,get_ingestion_settings
from rag_core.ingestion.scanner import  SourceAssetScanner
from rag_core.store.storage.provider import SourceProvider
from rag_core.store.postgres.session import Database
from rag_core.store.postgres.repositories.source_assets import SourceAssetRepository
from rag_core.store.postgres.repositories.ingestion_runs import IngestionRunRepository
from rag_core.ingestion.eligibility import EligibilityDecision
from rag_core.store.postgres.models.ingestion import AssetState, SourceAsset, IngestionRunStep, IngestionRunState
from rag_core.ingestion.eligibility import IngestionEligibilityService
from rag_core.ingestion.metadata import extract_metadata_from_uri
from rag_core.store.storage.provider import LocalFileSourceProvider
from rag_core.ingestion.scanner import FileScanner
from rag_core.store.postgres.session import get_database

logger = logging.getLogger(__name__)

class RunMetrics(BaseModel):
    """Strongly typed state tracker for pipeline execution."""
    discovered_total: int = Field(default=0)
    queued_new: int = Field(default=0)
    queued_reprocess: int = Field(default=0)
    skipped_unchanged: int = Field(default=0)
    failed: int = Field(default=0)
    deleted:int = Field(default=0)


class SourceDiscoveryService:
    def __init__(self,
        settings:IngestionSettings,
        scanner:SourceAssetScanner,
        source_peovider: SourceProvider,
        db:Database ) -> None:
        self.settings = settings
        self.source_provider= source_peovider
        self.scanner = scanner
        self.db = db
        self.run_id =None
        # self.metrics= RunMetrics()
    def run(self) ->RunMetrics |None:

        try:
            self._initialize()
            metrics= self._execute_scan_loop()
            deleted_asset_count=self._reconcile_source_assets()
            metrics.deleted+=deleted_asset_count
            self._transition_to_ingestion(metrics)

            return metrics

        except Exception as e:
            logger.exception("Catastrophic failure in ingestion pipeline run_id=%s, reason:%s", self.run_id, str(e))
            self._handle_failure(str(e))
            return None

    def _initialize(self) ->None:
        with self.db.get_session() as session:
             with session.begin():
                repo = IngestionRunRepository(session)
                run = repo.create_run()
                self.run_id = run.id
                logger.info("Created ingestion run run_id=%s", run.id)

    def _execute_scan_loop(self) ->RunMetrics:
        """The core work loop. Raises exceptions if something catastrophic happens."""

        self._update_progress(
            step=IngestionRunStep.SCANNING
        )
        metrics=RunMetrics()
        with self.db.get_session() as session:
            asset_repo = SourceAssetRepository(session)
            eligibility_service = IngestionEligibilityService(
                asset_repo,
                self.source_provider)

            for file_uri in self.scanner.discover_assets(self.settings.base_source_uri):
                metrics.discovered_total += 1
                current_asset_pk = None
                try:
                    logger.debug(f"Evaluating file: {file_uri}")
                    eval_result = eligibility_service.evaluate_file(file_uri)
                    current_asset_pk = eval_result.existing_asset_pk
                    logger.debug(
                        "Eligibility evaluated for file_uri=%s decision=%s should_process=%s existing_asset_pk=%s",
                        file_uri,
                        eval_result.decision.name,
                        eval_result.should_process,
                        eval_result.existing_asset_pk,
                    )
                    if not eval_result.should_process:
                        metrics.skipped_unchanged+=1
                        if eval_result.existing_asset_pk:
                            asset_repo.update_asset(
                                eval_result.existing_asset_pk,
                                last_seen_run_id = self.run_id
                            )
                            session.commit()
                        logger.debug(f"SKIPPED: {file_uri} - {eval_result.decision.name}")
                        continue

                    metadata = extract_metadata_from_uri(file_uri)

                    if eval_result.decision == EligibilityDecision.PROCESS_NEW:
                        metrics.queued_new += 1
                        asset = SourceAsset(
                            asset_key=eval_result.asset_key,
                            content_hash=eval_result.content_hash,
                            source_uri=self.source_provider.get_access_url(file_uri),
                            asset_name=metadata.file_name,
                            parent_name=metadata.folder_name,
                            tags=metadata.tags,
                            scopes=metadata.scopes,
                            asset_metadata=metadata.extended_metadata,
                            source_system=self.source_provider.source_system,
                            last_seen_run_id=self.run_id,
                            state=AssetState.PENDING
                        )
                        asset_repo.create_asset(asset)
                        logger.info(
                            "Queued new asset for processing file_uri=%s asset_key=%s",
                            file_uri,
                            eval_result.asset_key,
                        )
                    elif eval_result.decision == EligibilityDecision.REPROCESS_UPDATED:
                        metrics.queued_reprocess += 1
                        asset_repo.update_asset(
                            asset_id=eval_result.existing_asset_pk,
                            content_hash=eval_result.content_hash,
                            state=AssetState.PENDING,
                            last_seen_run_id=self.run_id,
                            updated_at=datetime.datetime.now(datetime.UTC),
                        )
                        logger.info(
                            "Queued existing asset for reprocessing file_uri=%s asset_pk=%s",
                            file_uri,
                            eval_result.existing_asset_pk,
                        )

                    session.commit()

                except Exception as e:
                    metrics.failed += 1
                    session.rollback()  # Clear any broken transactions so the loop can continue
                    logger.exception("Ingestion failed for file_uri=%s, Reason:%s", file_uri, str(e))
                    # If we know the asset ID, we can safely mark it as FAILED in the DB.
                    # If it failed before we got an ID, we just log it and move on.
                    if current_asset_pk:
                        try:
                            asset_repo.update_asset(
                                asset_id=current_asset_pk,
                                state=AssetState.FAILED,
                                last_seen_run_id=self.run_id
                            )
                            session.commit()
                        except Exception as secondary_e:
                            session.rollback()
                            logger.error(f"Failed to update asset state to FAILED: {secondary_e}")

        logger.info("Scan loop completed run_id=%s metrics=%s", self.run_id, metrics.model_dump())
        return metrics

    def _reconcile_source_assets(self) -> int:
        """Handles the transactional logic for soft-deleting absent assets."""
        logger.info("Starting reconciliation for system: %s", self.source_provider.source_system)
        self._update_progress(
            step=IngestionRunStep.RECONCILING
        )
        with self.db.get_session() as session:
            with session.begin():
                repo = SourceAssetRepository(session)
                affected_count = repo.mark_deleted_absent_assets(
                    source_system=self.source_provider.source_system,
                    current_run_id=self.run_id
                )

        logger.info("Reconciliation completed run_id=%s, deleted=%d", self.run_id, affected_count)
        return affected_count

    def _update_progress(self,step:IngestionRunStep,state:IngestionRunState=IngestionRunState.RUNNING, metrics:dict| None =None):
        with self.db.get_session() as session:
            with session.begin():
                runs_repo = IngestionRunRepository(session)
                runs_repo.update_progress(
                    self.run_id,
                    step=step,
                    state=state,
                    metrics=metrics
                )
                logger.info("Progress set run_id=%s, step=%s, state=%s",self.run_id,step,state )

    def _transition_to_ingestion(self, metrics: RunMetrics):
        """Signals that Discovery is done and the Worker can take over."""
        self._update_progress(
            # We move the baton to INGESTING so the Worker knows it's its turn
            step=IngestionRunStep.INGESTING,
            state=IngestionRunState.RUNNING,
            metrics=metrics.model_dump()
        )
        logger.info("Discovery phase complete. Run %s now awaiting worker.", self.run_id)
    def _handle_failure(self,error:str):
        with self.db.get_session() as session:
           with session.begin():
               repo = IngestionRunRepository(session)
               repo.fail(self.run_id, str(error))
               logger.info("Marked ingestion run failed run_id=%s", self.run_id)

def run_discovery() -> RunMetrics | None:
      settings = get_ingestion_settings()
      provider = LocalFileSourceProvider()
      scanner = FileScanner(provider,settings)
      db = get_database()

      service = SourceDiscoveryService(
          settings= settings,
        scanner=scanner,
        source_peovider=provider,
        db=db
      )
      return service.run()

if __name__=='__main__':
    run_discovery()
