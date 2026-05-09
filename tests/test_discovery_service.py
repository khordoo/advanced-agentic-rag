from __future__ import annotations

from contextlib import contextmanager
from io import BytesIO
import logging
from pathlib import Path
from typing import Any

from pytest import LogCaptureFixture, MonkeyPatch

from rag_core.core.config import IngestionSettings
from rag_core.ingestion.eligibility import (
    EligibilityDecision,
    EligibilityResult,
    IngestionEligibilityService,
)
from rag_core.ingestion.services.discovery import (
    DiscoveryRunMetrics,
    SourceDiscoveryService,
)
from rag_core.store.postgres.models.ingestion import (
    AssetState,
    IngestionRunStep,
    SourceAsset,
)
from rag_core.store.postgres.repositories.source_assets import SourceAssetRepository


class _FakeProvider:
    source_system = "test_fs"

    @contextmanager
    def open_stream(self, uri: str):
        del uri
        yield BytesIO(b"content")

    def get_access_url(self, uri: str) -> str:
        return f"access://{uri}"


class _FakeAssetRepo:
    def __init__(self) -> None:
        self.created: list[SourceAsset] = []
        self.updated: list[tuple[int, dict[str, Any]]] = []
        self.requeued: list[tuple[int, str, int]] = []

    def create_asset(self, asset: SourceAsset) -> SourceAsset:
        self.created.append(asset)
        return asset

    def update_asset(self, asset_id: int, **kwargs: Any) -> None:
        self.updated.append((asset_id, kwargs))

    def queue_asset_for_reprocessing(
        self,
        asset_id: int,
        content_hash: str,
        run_id: int,
    ) -> None:
        self.requeued.append((asset_id, content_hash, run_id))


class _NestedTransaction:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False


class _FakeSession:
    def __init__(self) -> None:
        self.begin_count = 0
        self.flush_count = 0
        self.begin_nested_count = 0

    def begin(self) -> _NestedTransaction:
        self.begin_count += 1
        return _NestedTransaction()

    def begin_nested(self) -> _NestedTransaction:
        self.begin_nested_count += 1
        return _NestedTransaction()

    def flush(self) -> None:
        self.flush_count += 1


class _FakeEligibilityService:
    def evaluate_file(self, asset_uri: str) -> EligibilityResult:
        del asset_uri
        return EligibilityResult(
            decision=EligibilityDecision.REPROCESS_UPDATED,
            existing_asset_pk=99,
            content_hash=None,
        )


class _FakeDbSessionScope:
    def __init__(self, session: _FakeSession) -> None:
        self.session = session

    def __enter__(self) -> _FakeSession:
        return self.session

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False


class _FakeDb:
    def __init__(self) -> None:
        self.session = _FakeSession()

    def get_session(self) -> _FakeDbSessionScope:
        return _FakeDbSessionScope(self.session)


def _build_service() -> SourceDiscoveryService:
    service = SourceDiscoveryService(
        settings=IngestionSettings(base_source_uri="source", discovery_batch_size=2),
        scanner=object(),  # type: ignore[arg-type]
        source_provider=_FakeProvider(),  # type: ignore[arg-type]
        db=object(),  # type: ignore[arg-type]
    )
    service.run_id = 42
    return service


def _build_asset(
    *,
    state: AssetState,
    content_hash: str = "old-hash",
    asset_id: int = 10,
) -> SourceAsset:
    return SourceAsset(
        id=asset_id,
        asset_key="asset-key",
        content_hash=content_hash,
        state=state,
        asset_name="policy.md",
        source_uri="source/legal/policy.md",
    )


def _evaluate_existing_asset(
    monkeypatch: MonkeyPatch,
    asset: SourceAsset,
    discovered_hash: str,
) -> EligibilityResult:
    class _AssetRepository:
        def get_asset(self, asset_key: str) -> SourceAsset:
            assert asset_key == "asset-key"
            return asset

    monkeypatch.setattr(
        "rag_core.ingestion.eligibility.generate_asset_key",
        lambda source_uri: "asset-key",
    )
    monkeypatch.setattr(
        "rag_core.ingestion.eligibility.compute_content_hash",
        lambda stream_manager, algorithm: discovered_hash,
    )

    service = IngestionEligibilityService(
        repo=_AssetRepository(),  # type: ignore[arg-type]
        storage_provider=_FakeProvider(),  # type: ignore[arg-type]
        hash_algorithm="sha256",
    )
    return service.evaluate_file("source/legal/policy.md")


def test_iter_batches_yields_full_and_partial_batches() -> None:
    batches = SourceDiscoveryService._iter_batches((str(i) for i in range(5)), 2)

    assert list(batches) == [["0", "1"], ["2", "3"], ["4"]]
    assert list(SourceDiscoveryService._iter_batches([], 2)) == []


def test_process_new_asset_creates_pending_source_asset(tmp_path: Path) -> None:
    asset_path = tmp_path / "finance" / "report.md"
    asset_path.parent.mkdir()
    asset_path.write_text("# report", encoding="utf-8")
    repo = _FakeAssetRepo()
    metrics = DiscoveryRunMetrics()

    _build_service()._apply_eligibility_decision(
        asset_uri=str(asset_path),
        eval_result=EligibilityResult.process_new(
            asset_key="finance_report_md",
            content_hash="hash123",
        ),
        asset_repo=repo,  # type: ignore[arg-type]
        metrics=metrics,
    )

    assert metrics.queued_new == 1
    assert metrics.queued_reprocess == 0
    assert metrics.skipped_unchanged == 0
    assert len(repo.created) == 1

    asset = repo.created[0]
    assert asset.asset_key == "finance_report_md"
    assert asset.content_hash == "hash123"
    assert asset.source_uri == f"access://{asset_path}"
    assert asset.asset_name == "report.md"
    assert asset.parent_name == "finance"
    assert asset.source_system == "test_fs"
    assert asset.last_seen_run_id == 42
    assert asset.state == AssetState.PENDING
    assert asset.tags == ["type:md", "folder:finance"]
    assert asset.scopes == ["finance_dept"]
    assert asset.asset_metadata == {"file_size_bytes": 8}


def test_hash_mismatch_requeues_failed_asset(monkeypatch: MonkeyPatch) -> None:
    result = _evaluate_existing_asset(
        monkeypatch,
        asset=_build_asset(state=AssetState.FAILED, content_hash="old-hash"),
        discovered_hash="new-hash",
    )

    assert result.decision == EligibilityDecision.REPROCESS_UPDATED
    assert result.existing_asset_pk == 10
    assert result.content_hash == "new-hash"


def test_hash_mismatch_requeues_processing_asset(monkeypatch: MonkeyPatch) -> None:
    result = _evaluate_existing_asset(
        monkeypatch,
        asset=_build_asset(state=AssetState.PROCESSING, content_hash="old-hash"),
        discovered_hash="new-hash",
    )

    assert result.decision == EligibilityDecision.REPROCESS_UPDATED
    assert result.existing_asset_pk == 10
    assert result.content_hash == "new-hash"


def test_hash_mismatch_does_not_override_excluded_asset(
    monkeypatch: MonkeyPatch,
) -> None:
    result = _evaluate_existing_asset(
        monkeypatch,
        asset=_build_asset(state=AssetState.EXCLUDED, content_hash="old-hash"),
        discovered_hash="new-hash",
    )

    assert result.decision == EligibilityDecision.IGNORE_EXCLUDED
    assert result.existing_asset_pk == 10
    assert result.content_hash is None


def test_unchanged_failed_asset_is_not_requeued(monkeypatch: MonkeyPatch) -> None:
    result = _evaluate_existing_asset(
        monkeypatch,
        asset=_build_asset(state=AssetState.FAILED, content_hash="same-hash"),
        discovered_hash="same-hash",
    )

    assert result.decision == EligibilityDecision.IGNORE_UNCHANGED
    assert result.existing_asset_pk == 10
    assert result.content_hash is None


def test_reprocess_updated_asset_marks_existing_asset_pending() -> None:
    repo = _FakeAssetRepo()
    metrics = DiscoveryRunMetrics()

    _build_service()._apply_eligibility_decision(
        asset_uri="source/legal/policy.md",
        eval_result=EligibilityResult.reprocess_updated(
            existing_asset_pk=7,
            content_hash="new-hash",
        ),
        asset_repo=repo,  # type: ignore[arg-type]
        metrics=metrics,
    )

    assert metrics.queued_new == 0
    assert metrics.queued_reprocess == 1
    assert repo.created == []
    assert repo.updated == []
    assert repo.requeued == [(7, "new-hash", 42)]

def test_queue_updated_asset_resets_worker_retry_and_lease_state() -> None:
    class _RecordingRepository:
        def __init__(self) -> None:
            self.statement = None

        def execute(self, stmt):
            self.statement = stmt

            class _Result:
                def scalar_one_or_none(self):
                    return None

            return _Result()

    session = _RecordingRepository()
    repo = SourceAssetRepository(session)  # type: ignore[arg-type]

    repo.queue_asset_for_reprocessing(
        asset_id=7,
        content_hash="new-hash",
        run_id=42,
    )

    assert session.statement is not None
    values = session.statement._values
    assert values[SourceAsset.__table__.c.content_hash].value == "new-hash"
    assert values[SourceAsset.__table__.c.state].value == AssetState.PENDING
    assert values[SourceAsset.__table__.c.attempt_count].value == 0
    assert values[SourceAsset.__table__.c.worker_id].value is None
    assert values[SourceAsset.__table__.c.leased_until].value is None
    assert values[SourceAsset.__table__.c.error_message].value is None
    assert values[SourceAsset.__table__.c.last_seen_run_id].value == 42
    assert "lease_version + " in str(values[SourceAsset.__table__.c.lease_version])


def test_unchanged_asset_is_marked_seen_without_queueing() -> None:
    repo = _FakeAssetRepo()
    metrics = DiscoveryRunMetrics()

    _build_service()._apply_eligibility_decision(
        asset_uri="source/legal/policy.md",
        eval_result=EligibilityResult.skip_unchanged(existing_asset_pk=11),
        asset_repo=repo,  # type: ignore[arg-type]
        metrics=metrics,
    )

    assert metrics.skipped_unchanged == 1
    assert metrics.queued_new == 0
    assert metrics.queued_reprocess == 0
    assert repo.created == []
    assert repo.updated == [(11, {"last_seen_run_id": 42})]


def test_asset_processing_failure_marks_existing_asset_failed() -> None:
    repo = _FakeAssetRepo()
    session = _FakeSession()
    metrics = DiscoveryRunMetrics()

    _build_service()._process_asset_with_savepoint(
        asset_uri="source/legal/policy.md",
        asset_repo=repo,  # type: ignore[arg-type]
        eligibility_service=_FakeEligibilityService(),  # type: ignore[arg-type]
        metrics=metrics,
        session=session,
    )

    assert metrics.discovered_total == 1
    assert metrics.failed == 1
    assert repo.updated == [(99, {"state": AssetState.FAILED, "last_seen_run_id": 42})]
    assert session.begin_nested_count == 2
    assert session.flush_count == 1


def test_asset_processing_failure_without_pk_skips_db_update() -> None:
    class _CrashingEligibilityService:
        def evaluate_file(self, asset_uri: str) -> EligibilityResult:
            del asset_uri
            raise PermissionError("Access denied")

    repo = _FakeAssetRepo()
    session = _FakeSession()
    metrics = DiscoveryRunMetrics()

    _build_service()._process_asset_with_savepoint(
        asset_uri="source/corrupt_file.pdf",
        asset_repo=repo,  # type: ignore[arg-type]
        eligibility_service=_CrashingEligibilityService(),  # type: ignore[arg-type]
        metrics=metrics,
        session=session,
    )

    assert metrics.discovered_total == 1
    assert metrics.failed == 1
    assert repo.updated == []
    assert session.begin_nested_count == 1
    assert session.flush_count == 0


def test_double_fault_during_failure_marking_is_swallowed_safely() -> None:
    class _DoubleFaultRepo(_FakeAssetRepo):
        def update_asset(self, asset_id: int, **kwargs: Any) -> None:
            if kwargs.get("state") == AssetState.FAILED:
                raise RuntimeError("DB connection lost")
            super().update_asset(asset_id, **kwargs)

    repo = _DoubleFaultRepo()
    session = _FakeSession()
    metrics = DiscoveryRunMetrics()

    _build_service()._process_asset_with_savepoint(
        asset_uri="source/legal/policy.md",
        asset_repo=repo,  # type: ignore[arg-type]
        eligibility_service=_FakeEligibilityService(),  # type: ignore[arg-type]
        metrics=metrics,
        session=session,
    )

    assert metrics.discovered_total == 1
    assert metrics.failed == 1
    assert repo.updated == []
    assert session.begin_nested_count == 2
    assert session.flush_count == 0


def test_unhandled_processable_eligibility_decision_logs_warning_and_skips(
    caplog: LogCaptureFixture,
) -> None:
    class _FakeDecision:
        name = "FUTURE_PROCESSABLE_STATE"

    class _FutureEligibilityResult(EligibilityResult):
        @property
        def should_process(self) -> bool:
            return True

    repo = _FakeAssetRepo()
    metrics = DiscoveryRunMetrics()
    eval_result = _FutureEligibilityResult(
        decision=_FakeDecision(),  # type: ignore[arg-type]
        existing_asset_pk=None,
        content_hash=None,
    )

    with caplog.at_level(logging.WARNING, logger="rag_core.ingestion.services.discovery"):
        _build_service()._apply_eligibility_decision(
            asset_uri="source/unknown.ext",
            eval_result=eval_result,
            asset_repo=repo,  # type: ignore[arg-type]
            metrics=metrics,
        )

    assert "Unhandled eligibility decision" in caplog.text
    assert "FUTURE_PROCESSABLE_STATE" in caplog.text
    assert metrics.queued_new == 0
    assert metrics.queued_reprocess == 0
    assert metrics.skipped_unchanged == 0
    assert repo.created == []
    assert repo.updated == []


def test_reconcile_source_assets_marks_absent_assets_for_current_run(
    monkeypatch: MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}

    class _ReconcilingAssetRepo:
        def __init__(self, session: _FakeSession) -> None:
            calls["session"] = session

        def mark_deleted_absent_assets(
            self,
            source_system: str,
            current_run_id: int,
        ) -> int:
            calls["source_system"] = source_system
            calls["current_run_id"] = current_run_id
            return 3

    service = _build_service()
    fake_db = _FakeDb()
    progress_updates: list[dict[str, Any]] = []
    service.db = fake_db  # type: ignore[assignment]
    service._update_progress = lambda **kwargs: progress_updates.append(kwargs)  # type: ignore[method-assign]
    monkeypatch.setattr(
        "rag_core.ingestion.services.discovery.SourceAssetRepository",
        _ReconcilingAssetRepo,
    )

    deleted = service._reconcile_source_assets()

    assert deleted == 3
    assert calls == {
        "session": fake_db.session,
        "source_system": "test_fs",
        "current_run_id": 42,
    }
    assert progress_updates == [{"step": IngestionRunStep.RECONCILING}]
    assert fake_db.session.begin_count == 1
