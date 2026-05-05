from typing import Any
import datetime
from sqlalchemy.orm import Session
from sqlalchemy import select,update
from rag_core.store.postgres.models.ingestion import AssetState, SourceAsset


class SourceAssetRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_next_pending(self):
        return (
            self.session.query(SourceAsset)
            .filter(SourceAsset.state == AssetState.PENDING)
            .with_for_update(skip_locked=True)
            .first()
        )
    def get_asset(self,asset_key:str) ->SourceAsset | None:
        stmt =(
            select(SourceAsset)
            .where(SourceAsset.asset_key == asset_key)
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def update_asset(self,asset_id:int,**kwargs:Any) ->SourceAsset | None:
        stmt =(
             update(SourceAsset)
            .where(SourceAsset.id ==asset_id)
            .values(**kwargs)
            .returning(SourceAsset)
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def create_asset(self,asset:SourceAsset) ->SourceAsset | None:
        return self.session.add(asset)

    def mark_deleted_absent_assets(self,source_system,current_run_id) -> int:
        """
            Soft-deletes assets that belong to a system but weren't seen in the current run.
        """
        stmt = (
            update(SourceAsset)
            .where(
                SourceAsset.source_system == source_system,
                SourceAsset.last_seen_run_id != current_run_id,
                SourceAsset.state.not_in([
                    AssetState.DELETED,
                    AssetState.EXCLUDED,
                ])
             )
            .values(
                state=AssetState.DELETED,
                updated_at = datetime.datetime.now(tz=datetime.UTC),
            )
        )
        result = self.session.execute(stmt)
        return result.rowcount or 0
