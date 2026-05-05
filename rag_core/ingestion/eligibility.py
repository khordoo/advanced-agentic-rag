import logging
import enum
from dataclasses import dataclass
from typing import Self

from rag_core.core.config import get_ingestion_settings
from rag_core.ingestion.fingerprinting import compute_content_hash, generate_asset_key
from rag_core.store.postgres.models.ingestion import AssetState
from rag_core.store.postgres.repositories.source_assets import SourceAssetRepository
from rag_core.store.storage.provider import LocalFileSourceProvider, SourceProvider

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
    asset_key: str | None = None
    content_hash: str | None = None
    existing_asset_pk: int | None = None

    @property
    def should_process(self) -> bool:
        return self.decision in {
            EligibilityDecision.PROCESS_NEW,
            EligibilityDecision.REPROCESS_UPDATED
        }

    @classmethod
    def process_new(cls, asset_key: str, content_hash: str) -> Self:
        """Use when the asset key is not found in the DB."""
        return cls(
            decision=EligibilityDecision.PROCESS_NEW,
            asset_key=asset_key,
            content_hash=content_hash,
        )

    @classmethod
    def reprocess_updated(cls, existing_asset_pk: int, content_hash: str) -> Self:
        """Use when the asset key exists but the content hash changed."""
        return cls(
            decision=EligibilityDecision.REPROCESS_UPDATED,
            existing_asset_pk=existing_asset_pk,
            content_hash=content_hash,
        )

    @classmethod
    def skip_unchanged(cls, existing_asset_pk: int) -> Self:
        """Use when key and hash both match the database."""
        return cls(
            decision=EligibilityDecision.IGNORE_UNCHANGED,
            existing_asset_pk=existing_asset_pk,
        )



class IngestionEligibilityService:
    def __init__(self,repo: SourceAssetRepository, storage_provider: SourceProvider, hash_algorithm:str='sha256') -> None:
        self.repo = repo
        self.storage_provider = storage_provider or LocalFileSourceProvider()
        self.hash_algorithm = hash_algorithm or get_ingestion_settings().hash_algorithm

    def evaluate_file(self,source_uri: str) -> EligibilityResult:
        asset_key = generate_asset_key(source_uri)
        asset=self.repo.get_asset(asset_key)
        stream_manager = self.storage_provider.open_stream(source_uri)
        content_hash = compute_content_hash(stream_manager, algorithm=self.hash_algorithm)
        if not asset:
            logger.info("Asset key does not exist. Queuing as new asset. asset_key=%s", asset_key)
            return EligibilityResult.process_new(asset_key=asset_key, content_hash=content_hash)
        elif asset.state ==AssetState.EXCLUDED:
           return EligibilityResult(
               decision=EligibilityDecision.IGNORE_EXCLUDED,
               asset_key=asset_key,
               existing_asset_pk=asset.id,
           )
        elif content_hash != asset.content_hash:
            return EligibilityResult.reprocess_updated(
                existing_asset_pk=asset.id,
                content_hash=content_hash,
            )
        elif content_hash == asset.content_hash:
            return EligibilityResult.skip_unchanged(existing_asset_pk=asset.id)

        logger.warning(
            "Asset eligibility could not be determined. Skipped as unsupported. asset_key=%s",
            asset_key,
        )
        return EligibilityResult(
            decision=EligibilityDecision.IGNORE_UNSUPPORTED,
            asset_key=asset_key,
            content_hash=content_hash,
            existing_asset_pk=asset.id,
        )
