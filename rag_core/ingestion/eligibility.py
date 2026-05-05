import enum
import logging
from pathlib import Path
from dataclasses import dataclass
from rag_core.store.postgres.repositories.source_assets import SourceAssetRepository
from rag_core.ingestion.fingerprinting import generate_file_id, compute_content_hash
from rag_core.store.storage.provider import SourceProvider, LocalFileSourceProvider
from rag_core.store.postgres.models.ingestion import AssetState
from rag_core.core.config import get_ingestion_settings

logger = logging.getLogger(__name__)

class EligibilityDecision(str, enum.Enum):
    PROCESS_NEW = "PROCESS_NEW"
    REPROCESS_UPDATED = "REPROCESS_UPDATED"
    IGNORE_UNCHANGED = "IGNORE_UNCHANGED"
    IGNORE_UNSUPPORTED = "IGNORE_UNSUPPORTED"
    IGNORE_EXCLUDED="IGNORE_EXCLUDED"



@dataclass
class EligibilityResult:
    decision: EligibilityDecision
    file_id: str | None =None
    content_hash: str | None = None
    existing_asset_id: int | None = None

    @property
    def should_process(self) -> bool:
        return self.decision in {
            EligibilityDecision.PROCESS_NEW,
            EligibilityDecision.REPROCESS_UPDATED
        }



class IngestionEligibilityService:
    def __init__(self,repo: SourceAssetRepository, storage_provider: SourceProvider, hash_algorithm:str='sha256') -> None:
        self.repo = repo
        self.storage_provider = storage_provider or LocalFileSourceProvider()
        self.hash_algorithm = hash_algorithm or get_ingestion_settings().hash_algorithm
    def evaluate_file(self,source_uri: str) -> EligibilityResult:
        file_id = generate_file_id(source_uri)
        asset=self.repo.get_asset(file_id)
        stream_manager = self.storage_provider.open_stream(source_uri)
        content_hash = compute_content_hash(stream_manager, algorithm=self.hash_algorithm)
        decision = EligibilityDecision.IGNORE_UNSUPPORTED
        if not asset:
            print('File does nto exist creasting new ingestion file')
            decision = EligibilityDecision.PROCESS_NEW
        elif asset.state ==AssetState.EXCLUDED:
           decision = EligibilityDecision.IGNORE_EXCLUDED
        elif content_hash != asset.content_hash:
            decision = EligibilityDecision.REPROCESS_UPDATED
        elif content_hash == asset.content_hash:
            decision = EligibilityDecision.IGNORE_UNCHANGED
        else:
            decision = EligibilityDecision.IGNORE_UNSUPPORTED
            logger.warning("File eligiblity could not be determined. skipped with Unsupported flag!. file_id :%s",file_id)

        return EligibilityResult(decision=decision,
            file_id=file_id,
            content_hash=content_hash,
            existing_asset_id= asset.id if asset else None
        )
