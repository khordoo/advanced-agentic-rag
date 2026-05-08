from typing import Any
import datetime
from chromadb.api.models.CollectionCommon import Where
from sqlalchemy.orm import Session
from sqlalchemy import select,update, func, text,or_, and_, case
from sqlalchemy.sql.expression import values
from rag_core.store.postgres.models.ingestion import AssetState, SourceAsset


class SourceAssetRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def claim_next_asset(self,worker_id:str,max_attempts:int,lease_duration_seconds:int)->SourceAsset | None:
        asset_id_subquery = (
            select(SourceAsset.id)
            .where(
                or_(
                    SourceAsset.state == AssetState.PENDING,
                    and_(
                        SourceAsset.state == AssetState.PROCESSING,
                        SourceAsset.leased_until < func.now(),
                    )

                ),
            )
            .where(SourceAsset.attempt_count <max_attempts)
            .order_by(SourceAsset.id)
            .with_for_update(skip_locked=True)
            .order_by(SourceAsset.id)
            .limit(1)
            .scalar_subquery()
        )
        lease_expiration = lease_expiration = func.now() + text(f"interval '{lease_duration_seconds} seconds'")
        stmt = (
            update(SourceAsset)
            .where(SourceAsset.id==asset_id_subquery)
            .values(
                state=AssetState.PROCESSING,
                worker_id=worker_id,
                leased_until=lease_expiration,
                updated_at=func.now(),
                attempt_count= SourceAsset.attempt_count+1
            )
            .returning(SourceAsset)

        )
        return self.session.execute(stmt).scalar_one_or_none()


    def attempt_mark_asset_complete(self,worker_id:str,asset_id:int)->bool:
        """
            Attempts to mark an asset as completed.
            Returns True if successful, False if the worker lost the lease.
            """
        stmt=(
            update(SourceAsset)
            .where(
                SourceAsset.id == asset_id,
                SourceAsset.worker_id == worker_id,
                SourceAsset.state == AssetState.PROCESSING
            )
            .values(
                state=AssetState.COMPLETED,
                worker_id=None,
                leased_until=None,
                updated_at=func.now()
            )
        )
        result = self.session.execute(stmt)
        return result.rowcount==1


    def extend_lease(self, worker_id: str, asset_id: int, extra_seconds: int = 60) -> bool:
            """
            Attempts to extend the lease.
            Returns True if successful, False if the worker lost the lease.
            """
            stmt = (
                update(SourceAsset)
                .where(
                    SourceAsset.id == asset_id,
                    SourceAsset.worker_id == worker_id,
                    SourceAsset.state == AssetState.PROCESSING
                )
                .values(
                    leased_until=func.now() + text(f"interval '{extra_seconds} seconds'"),
                    updated_at=func.now()
                )
            )
            return self.session.execute(stmt).rowcount == 1

    def conditional_mark_failed(self, worker_id: str, asset_id: int, error_message: str, max_attempts: int) -> bool:
            """
            Releases a failed asset.
            If it has retries left, it goes back to PENDING.
            If it is out of retries, it becomes permanently FAILED.
            """
            stmt = (
                update(SourceAsset)
                .where(
                    SourceAsset.id == asset_id,
                    SourceAsset.worker_id == worker_id,
                    SourceAsset.state == AssetState.PROCESSING
                )
                .values(
                    #The Smart State Switch
                    state=case(
                        (SourceAsset.attempt_count >= max_attempts, AssetState.FAILED),
                        else_=AssetState.PENDING
                    ),
                    worker_id=None,
                    leased_until=None,
                    error_message=error_message,
                    updated_at=func.now()
                )
            )
            return self.session.execute(stmt).rowcount == 1

    def mark_poison_pills_as_failed(self, max_attempt_count: int) -> int:
        """
        Sweeps the table for assets that crashed workers too many times
        and permanently marks them as FAILED.
        """
        stmt = (
            update(SourceAsset)
            .where(
                # Catch assets that are stuck in PROCESSING but out of retries
                SourceAsset.attempt_count >= max_attempt_count,
                SourceAsset.state.in_([AssetState.PENDING, AssetState.PROCESSING])
            )
            .values(
                state=AssetState.FAILED,
                worker_id=None,
                updated_at=func.now(),
                error_message="Asset exceeded maximum worker attempt count (Poison Pill)."
            )
        )

        result = self.session.execute(stmt)
        return result.rowcount # Returns how many files were permanently failed

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
