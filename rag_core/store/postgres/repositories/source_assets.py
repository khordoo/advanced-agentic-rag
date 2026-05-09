import datetime
from typing import Any

from sqlalchemy import and_, case, func, or_, select, text, update
from sqlalchemy.sql.elements import TextClause
from sqlalchemy.orm import Session

from rag_core.schemas import ClaimedSourceAsset
from rag_core.store.postgres.models.ingestion import AssetState, SourceAsset


def _postgres_interval_seconds(seconds: int) -> TextClause:
    if seconds <= 0:
        raise ValueError("seconds must be greater than zero")
    return text(f"interval '{seconds} seconds'")


class SourceAssetRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def claim_next_asset(
        self,
        worker_id: str,
        max_attempts: int,
        lease_duration_seconds: int,
    ) -> ClaimedSourceAsset | None:
        asset_id_subquery = (
            select(SourceAsset.id)
            .where(
                or_(
                    SourceAsset.state == AssetState.PENDING,
                    and_(
                        SourceAsset.state == AssetState.PROCESSING,
                        SourceAsset.leased_until < func.now(),
                    ),
                ),
            )
            .where(SourceAsset.attempt_count < max_attempts)
            .with_for_update(skip_locked=True)
            .order_by(SourceAsset.id)
            .limit(1)
            .scalar_subquery()
        )
        lease_expiration = func.now() + _postgres_interval_seconds(
            lease_duration_seconds
        )
        stmt = (
            update(SourceAsset)
            .where(SourceAsset.id == asset_id_subquery)
            .values(
                state=AssetState.PROCESSING,
                worker_id=worker_id,
                lease_version=SourceAsset.lease_version + 1,
                leased_until=lease_expiration,
                updated_at=func.now(),
                attempt_count=SourceAsset.attempt_count + 1,
            )
            .returning(
                SourceAsset.id,
                SourceAsset.asset_key,
                SourceAsset.asset_name,
                SourceAsset.parent_name,
                SourceAsset.source_system,
                SourceAsset.source_uri,
                SourceAsset.tags,
                SourceAsset.scopes,
                SourceAsset.asset_metadata,
                SourceAsset.lease_version,
            )
        )
        row = self.session.execute(stmt).mappings().one_or_none()
        if row is None:
            return None

        return ClaimedSourceAsset(
            id=row["id"],
            asset_key=row["asset_key"],
            asset_name=row["asset_name"],
            parent_name=row["parent_name"],
            source_system=row["source_system"],
            source_uri=row["source_uri"],
            tags=list(row["tags"] or []),
            scopes=list(row["scopes"] or []),
            asset_metadata=dict(row["asset_metadata"] or {}),
            lease_version=row["lease_version"],
        )

    def mark_asset_complete_if_owned(
        self, expected_version: int, asset_id: int
    ) -> bool:
        """
        Mark an asset completed when the caller's lease version still matches.

        Returns True when the row was updated, False when another worker
        claimed the asset (lease_version changed).
        """
        stmt = (
            update(SourceAsset)
            .where(
                SourceAsset.id == asset_id,
                SourceAsset.lease_version == expected_version,
                SourceAsset.state == AssetState.PROCESSING,
            )
            .values(
                state=AssetState.COMPLETED,
                worker_id=None,
                leased_until=None,
                updated_at=func.now(),
            )
        )
        result = self.session.execute(stmt)
        return result.rowcount == 1

    def extend_asset_lease(
        self,
        expected_version: int,
        asset_id: int,
        extra_seconds: int = 60,
    ) -> bool:
        """
        Extend an asset lease when the caller's lease version still matches.

        Returns True when the row was updated, False when another worker
        claimed the asset (lease_version changed).
        """
        stmt = (
            update(SourceAsset)
            .where(
                SourceAsset.id == asset_id,
                SourceAsset.lease_version == expected_version,
                SourceAsset.state == AssetState.PROCESSING,
            )
            .values(
                leased_until=func.now() + _postgres_interval_seconds(extra_seconds),
                updated_at=func.now(),
            )
        )
        return self.session.execute(stmt).rowcount == 1

    def mark_asset_failed_or_retry_if_owned(
        self,
        expected_version: int,
        asset_id: int,
        error_message: str,
        max_attempts: int,
    ) -> AssetState | None:
        """
        Release a failed asset when the caller's lease version still matches.

        If the asset has retries left, it returns to PENDING. If it is out of
        retries, it becomes permanently FAILED.

        Returns the new asset state, or None when another worker claimed the
        asset (lease_version changed).
        """
        stmt = (
            update(SourceAsset)
            .where(
                SourceAsset.id == asset_id,
                SourceAsset.lease_version == expected_version,
                SourceAsset.state == AssetState.PROCESSING,
            )
            .values(
                state=case(
                    (SourceAsset.attempt_count >= max_attempts, AssetState.FAILED),
                    else_=AssetState.PENDING,
                ),
                worker_id=None,
                leased_until=None,
                error_message=error_message,
                updated_at=func.now(),
            )
            .returning(SourceAsset.state)
        )
        return self.session.execute(stmt).scalar_one_or_none()

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
                SourceAsset.state.in_([AssetState.PENDING, AssetState.PROCESSING]),
            )
            .values(
                state=AssetState.FAILED,
                worker_id=None,
                updated_at=func.now(),
                error_message="Asset exceeded maximum worker attempt count (Poison Pill).",
            )
        )

        result = self.session.execute(stmt)
        return result.rowcount  # Returns how many files were permanently failed

    def get_asset(self, asset_key: str) -> SourceAsset | None:
        stmt = select(SourceAsset).where(SourceAsset.asset_key == asset_key)
        return self.session.execute(stmt).scalar_one_or_none()

    def update_asset(self, asset_id: int, **kwargs: Any) -> SourceAsset | None:
        stmt = (
            update(SourceAsset)
            .where(SourceAsset.id == asset_id)
            .values(**kwargs)
            .returning(SourceAsset)
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def queue_asset_for_reprocessing(
        self,
        asset_id: int,
        content_hash: str,
        run_id: int,
    ) -> SourceAsset | None:
        stmt = (
            update(SourceAsset)
            .where(SourceAsset.id == asset_id)
            .values(
                content_hash=content_hash,
                state=AssetState.PENDING,
                attempt_count=0,
                worker_id=None,
                leased_until=None,
                error_message=None,
                lease_version=SourceAsset.lease_version + 1,
                last_seen_run_id=run_id,
                updated_at=func.now(),
            )
            .returning(SourceAsset)
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def create_asset(self, asset: SourceAsset) -> SourceAsset:
        self.session.add(asset)
        return asset

    def mark_deleted_absent_assets(self, source_system, current_run_id) -> int:
        """
            Soft-deletes assets that belong to a system but weren't seen in the current run.
        """
        stmt = (
            update(SourceAsset)
            .where(
                SourceAsset.source_system == source_system,
                SourceAsset.last_seen_run_id != current_run_id,
                SourceAsset.state.not_in(
                    [
                        AssetState.DELETED,
                        AssetState.EXCLUDED,
                    ]
                ),
            )
            .values(
                state=AssetState.DELETED,
                updated_at=datetime.datetime.now(tz=datetime.UTC),
            )
        )
        result = self.session.execute(stmt)
        return result.rowcount or 0
