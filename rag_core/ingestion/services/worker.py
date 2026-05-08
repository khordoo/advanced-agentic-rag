import uuid
import socket
import logging
from rag_core.store.postgres.repositories.source_assets import SourceAssetRepository
from rag_core.store.postgres.session import Database, get_database
from rag_core.store.postgres.models.ingestion import AssetState, SourceAsset
from rag_core.core.config import get_ingestion_settings, IngestionSettings
from rag_core.indexing.base import BaseIndexer
from rag_core.ingestion.processing.base import BaseProcessor
from rag_core.indexing.qdrant_indexer import QdrantIndexer
import time
import random

logger = logging.getLogger(__name__)



class SourceIngestionWorkerService:
    def __init__(self,
        db: Database | None=None,
        settings:IngestionSettings |None=None,
        processor: BaseProcessor | None = None,
        indexer:BaseIndexer |None=None):
        self.db = db or get_database()
        self.settings = settings or get_ingestion_settings()
        self.indexer=indexer or QdrantIndexer()
        hostname = socket.gethostname()
        short_uuid = uuid.uuid4().hex[:8]
        self.worker_id = f"worker-{hostname}-{short_uuid}"


    def run(self):
        logger.info(f"Worker {self.worker_id} starting ingestion loop...")
        while (asset:=self._attempt_claim_asset()):
            try:
                logger.info(f"Worker {self.worker_id}, Porcessing asset: {asset.id}")
                chunks= self._process(asset)
                # THE PRE-FLIGHT CHECK
                # Give ourselves 60 guaranteed seconds to push to Qdrant.
                # If this returns False, we lost the lease during step 1. Abort!
                if not self._attempt_extend_lease(asset.id, extra_seconds=60):
                    print(f"Lost lease on asset {asset.id} during processing. Aborting save.")
                    continue # Drop the chunks, move to the next file

                self._save_result(chunks)
                # 4. FINALIZE (Postgres)
                if self._attempt_mark_complete(asset.id):
                    print(f"Asset {asset.id} fully completed.")
                else:
                    print(f"CRITICAL: Lost lease on {asset.id} AFTER Qdrant push.")
            except Exception as e:
                logger.error(f"Failure processing asset {asset.id}: {e}")
                self._attempt_conditional_mark_failed(asset.id, error_message=str(e))
        print('all asssts completed.')
    def _attempt_claim_asset(self) ->SourceAsset | None:
        with self.db.get_session() as session:
            asset=None
            with session.begin():
                repo = SourceAssetRepository(session)
                asset= repo.claim_next_asset(
                    worker_id=self.worker_id,
                    max_attempts=self.settings.worker_max_attempts,
                    lease_duration_seconds=self.settings.worker_lease_duration_seconds
                )
            print('Clained asset object:',asset)
            return asset

    def _attempt_mark_complete(self,asset_id:int):
        with self.db.get_session() as session:
            with session.begin():
                repo = SourceAssetRepository(session)
                return repo.attempt_mark_asset_complete(
                    worker_id=self.worker_id,
                    asset_id=asset_id
                )

    def _attempt_extend_lease(self,asset_id:int,extra_seconds=60):
        with self.db.get_session() as session:
            with session.begin():
                repo = SourceAssetRepository(session)
                return repo.extend_lease(
                    worker_id=self.worker_id,
                    asset_id=asset_id
                )
    def _attempt_conditional_mark_failed(self,asset_id:int,error_message:str):
        with self.db.get_session() as session:
            with session.begin():
                repo = SourceAssetRepository(session)
                return repo.conditional_mark_failed(
                    worker_id=self.worker_id,
                    asset_id=asset_id,
                    error_message=error_message,
                    max_attempts= self.settings.worker_max_attempts
                )
    def _process(self,asset:SourceAsset):
        print('processing asset:',asset.source_uri)
        wait_time =random.randint(3,6)
        time.sleep(wait_time)
        print("finished Pprocessong time=%d, source_url=%s",wait_time,asset.source_uri)

    def _save_result(self,chunks):
        print('exporrrting chukns to indexer to save in vector database')

SourceIngestionWorkerService().run()
