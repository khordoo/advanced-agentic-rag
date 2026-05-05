import logging
import datetime

from rag_core.core.config import get_ingestion_settings, load_config, setup_logging
from rag_core.ingestion.metadata import extract_metadata_from_uri
from rag_core.ingestion.scanner import FileScanner
from rag_core.ingestion.eligibility import IngestionEligibilityService
from rag_core.store.postgres.repositories.source_assets import SourceAssetRepository
from rag_core.store.postgres.repositories.ingestion_runs import IngestionRunRepository
from rag_core.ingestion.eligibility import EligibilityDecision,EligibilityResult
from rag_core.store.postgres.models.ingestion import AssetState, IngestionRun, SourceAsset
from rag_core.store.storage.provider import LocalFileSourceProvider
from rag_core.store.postgres.session import get_database
from rag_core.ingestion.reconciliation import reconcile_source_assets
from pydantic import BaseModel, Field

setup_logging()
logger = logging.getLogger(__name__)



class RunMetrics(BaseModel):
    """Strongly typed state tracker for pipeline execution."""
    discovered_total: int = Field(default=0)
    queued_new: int = Field(default=0)
    queued_reprocess: int = Field(default=0)
    skipped_unchanged: int = Field(default=0)
    failed: int = Field(default=0)
    deleted:int = Field(default=0)

def execute_scan_loop(session, run_id, settings, source_provider, scanner, eligibility_service, asset_repo) ->RunMetrics:
    """The core work loop. Raises exceptions if something catastrophic happens."""
    metrics=RunMetrics()
    for file_uri in scanner.discover_eligible_files(settings.base_source_uri):
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
                        last_seen_run_id = run_id
                    )
                    session.commit()
                logger.info(f"SKIPPED: {file_uri} - {eval_result.decision.name}")
                continue

            metadata = extract_metadata_from_uri(file_uri)

            if eval_result.decision == EligibilityDecision.PROCESS_NEW:
                metrics.queued_new += 1
                asset = SourceAsset(
                    asset_key=eval_result.asset_key,
                    content_hash=eval_result.content_hash,
                    source_uri=source_provider.get_access_url(file_uri),
                    asset_name=metadata.file_name,
                    parent_name=metadata.folder_name,
                    tags=metadata.tags,
                    scopes=metadata.scopes,
                    asset_metadata=metadata.extended_metadata,
                    source_system=source_provider.source_system,
                    last_seen_run_id=run_id,
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
                    last_seen_run_id=run_id,
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
            logger.exception("Ingestion failed for file_uri=%s", file_uri)
            # If we know the asset ID, we can safely mark it as FAILED in the DB.
            # If it failed before we got an ID, we just log it and move on.
            if current_asset_pk:
                try:
                    asset_repo.update_asset(
                        asset_id=current_asset_pk,
                        state=AssetState.FAILED,
                        last_seen_run_id=run_id
                    )
                    session.commit()
                except Exception as secondary_e:
                    session.rollback()
                    logger.error(f"Failed to update asset state to FAILED: {secondary_e}")

    return metrics



def run_pipeline():
    logger.info("Starting ingestion run pipeline")
    ingestion_run_id: int = None
    settings = get_ingestion_settings()
    source_provider= LocalFileSourceProvider()
    scanner = FileScanner(provider=source_provider,settings=settings)
    logger.info(
        "Ingestion settings loaded source_root=%s allowed_extensions=%s source_system=%s",
        settings.base_source_uri,
        settings.allowed_extensions,
        source_provider.source_system,
    )
    db= get_database()
    with db.get_session() as session:
         with session.begin():
            repo = IngestionRunRepository(session)
            ingestion_run = repo.create_run()
            ingestion_run_id = ingestion_run.id
            logger.info("Created ingestion run run_id=%s", ingestion_run_id)

    try:
        with db.get_session() as session:
            asset_repo = SourceAssetRepository(session)
            eligibility_service = IngestionEligibilityService(
                asset_repo,
                source_provider
            )
            metrics=execute_scan_loop(
            session, ingestion_run_id, settings, source_provider,
            scanner, eligibility_service, asset_repo
            )
        logger.info("Scan loop completed run_id=%s metrics=%s", ingestion_run_id, metrics.model_dump())
        logger.info("Starting reconciliation for system: %s", source_provider.source_system)
        deleted_asset_count = reconcile_source_assets(
            db_manager=db,
            source_system=source_provider.source_system,
            run_id=ingestion_run_id
        )
        metrics.deleted+=deleted_asset_count
        logger.info("Reconciliation completed run_id=%s, deleted=%d", ingestion_run_id, metrics.deleted)

        with db.get_session() as session:
            with session.begin():
                repo = IngestionRunRepository(session)
                ingestion_run = repo.mark_completed(ingestion_run_id,
                    metrics=metrics.model_dump())
                logger.info("Marked ingestion run completed run_id=%s", ingestion_run_id)
        logger.info("Ingestion run completed run_id=%s metrics=%s", ingestion_run_id, metrics.model_dump())
    except Exception as e:
       logger.exception("Catastrophic failure in ingestion pipeline run_id=%s", ingestion_run_id)
       with db.get_session() as session:
           with session.begin():
               repo = IngestionRunRepository(session)
               ingestion_run = repo.mark_failed(ingestion_run_id, str(e))
               logger.info("Marked ingestion run failed run_id=%s", ingestion_run_id)

if __name__ == "__main__":
    run_pipeline()
