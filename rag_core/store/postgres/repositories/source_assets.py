from typing import Any
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
    def get_asset(self,file_id:str) ->SourceAsset | None:
        stmt =(
            select(SourceAsset)
            .where(SourceAsset.file_id ==file_id)
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
