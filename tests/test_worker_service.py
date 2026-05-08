from __future__ import annotations

from dataclasses import FrozenInstanceError
import logging
from pathlib import Path

import pytest

from rag_core.core.config import IngestionSettings
from rag_core.indexing.base import BaseIndexer
from rag_core.ingestion.processing.base import BaseProcessor
from rag_core.ingestion.services import worker as worker_module
from rag_core.ingestion.services.worker import SourceIngestionWorkerService
from rag_core.schemas import ClaimedSourceAsset, ParentChunk
from rag_core.store.postgres.models.ingestion import AssetState
from rag_core.store.postgres.repositories.source_assets import SourceAssetRepository


def _settings(preflight_extension_seconds: int = 17) -> IngestionSettings:
    return IngestionSettings(
        worker_max_attempts=3,
        worker_lease_duration_seconds=300,
        worker_preflight_extension_seconds=preflight_extension_seconds,
    )


def _asset(asset_id: int = 1) -> ClaimedSourceAsset:
    return ClaimedSourceAsset(
        id=asset_id,
        asset_key=f"asset-{asset_id}",
        asset_name=f"asset-{asset_id}.md",
        parent_name="folder",
        source_system="local_fs",
        source_uri=f"file:///asset-{asset_id}.md",
        tags=["test"],
        scopes=["public"],
        asset_metadata={"asset_id": asset_id},
    )


class _FakeProcessor(BaseProcessor):
    def __init__(
        self,
        chunks: list[ParentChunk] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.chunks = chunks or []
        self.error = error
        self.processed_assets: list[ClaimedSourceAsset] = []

    def process_asset(self, asset: ClaimedSourceAsset) -> list[ParentChunk]:
        self.processed_assets.append(asset)
        if self.error is not None:
            raise self.error
        return self.chunks


class _FakeIndexer(BaseIndexer):
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.ingested_batches: list[list[ParentChunk]] = []

    def ingest_documents(self, document_blocks: list[ParentChunk]) -> None:
        if self.error is not None:
            raise self.error
        self.ingested_batches.append(document_blocks)

    def reset_collections(self) -> None:
        pass


class _WorkerHarness(SourceIngestionWorkerService):
    def __init__(
        self,
        assets: list[ClaimedSourceAsset],
        *,
        processor: _FakeProcessor | None = None,
        indexer: _FakeIndexer | None = None,
        settings: IngestionSettings | None = None,
        extend_result: bool = True,
        complete_result: bool = True,
        fail_mark_result: AssetState | None = AssetState.PENDING,
    ) -> None:
        self.claimable_assets = list(assets)
        self.extended_leases: list[tuple[int, int]] = []
        self.completed_assets: list[int] = []
        self.failed_assets: list[tuple[int, str]] = []
        self.extend_result = extend_result
        self.complete_result = complete_result
        self.fail_mark_result = fail_mark_result

        super().__init__(
            db=object(),
            settings=settings or _settings(),
            processor=processor or _FakeProcessor(),
            indexer=indexer or _FakeIndexer(),
            worker_id="worker-test",
        )

    def _claim_next_asset(self) -> ClaimedSourceAsset | None:
        if not self.claimable_assets:
            return None
        return self.claimable_assets.pop(0)

    def _extend_asset_lease(self, asset_id: int, extra_seconds: int) -> bool:
        self.extended_leases.append((asset_id, extra_seconds))
        return self.extend_result

    def _mark_asset_complete(self, asset_id: int) -> bool:
        self.completed_assets.append(asset_id)
        return self.complete_result

    def _mark_asset_failed(
        self,
        asset_id: int,
        error_message: str,
    ) -> AssetState | None:
        self.failed_assets.append((asset_id, error_message))
        return self.fail_mark_result


def test_worker_requires_processor_and_indexer() -> None:
    with pytest.raises(ValueError, match="requires both processor and indexer"):
        SourceIngestionWorkerService(settings=_settings(), processor=_FakeProcessor())

    with pytest.raises(ValueError, match="requires both processor and indexer"):
        SourceIngestionWorkerService(settings=_settings(), indexer=_FakeIndexer())


def test_default_preflight_extension_setting_is_sixty_seconds() -> None:
    field = IngestionSettings.model_fields["worker_preflight_extension_seconds"]
    assert field.default == 60


def test_claimed_source_asset_is_frozen() -> None:
    asset = _asset(5)

    with pytest.raises(FrozenInstanceError):
        asset.source_uri = "file:///changed.md"


def test_worker_module_uses_main_guard_for_script_execution() -> None:
    source = Path(worker_module.__file__).read_text(encoding="utf-8")

    assert 'if __name__ == "__main__":' in source


def test_claim_next_asset_returns_claimed_source_asset_snapshot() -> None:
    class _FakeResult:
        def mappings(self):
            return self

        def one_or_none(self):
            return {
                "id": 70,
                "asset_key": "asset-70",
                "asset_name": "asset-70.md",
                "parent_name": "finance",
                "source_system": "local_fs",
                "source_uri": "file:///asset-70.md",
                "tags": None,
                "scopes": None,
                "asset_metadata": None,
            }

    class _FakeSession:
        def execute(self, stmt):
            self.executed_stmt = stmt
            return _FakeResult()

    repo = SourceAssetRepository(_FakeSession())

    asset = repo.claim_next_asset(
        worker_id="worker-test",
        max_attempts=3,
        lease_duration_seconds=300,
    )

    assert asset == ClaimedSourceAsset(
        id=70,
        asset_key="asset-70",
        asset_name="asset-70.md",
        parent_name="finance",
        source_system="local_fs",
        source_uri="file:///asset-70.md",
        tags=[],
        scopes=[],
        asset_metadata={},
    )


def test_mark_asset_failed_or_retry_returns_new_state() -> None:
    class _FakeResult:
        def scalar_one_or_none(self):
            return AssetState.FAILED

    class _FakeSession:
        def execute(self, stmt):
            self.executed_stmt = stmt
            return _FakeResult()

    repo = SourceAssetRepository(_FakeSession())

    new_state = repo.mark_asset_failed_or_retry_if_owned(
        worker_id="worker-test",
        asset_id=80,
        error_message="boom",
        max_attempts=3,
    )

    assert new_state == AssetState.FAILED


def test_run_once_processes_indexes_and_marks_complete() -> None:
    processor = _FakeProcessor()
    indexer = _FakeIndexer()
    worker = _WorkerHarness([_asset(10)], processor=processor, indexer=indexer)

    assert worker.run_once() is True

    assert [asset.id for asset in processor.processed_assets] == [10]
    assert indexer.ingested_batches == [[]]
    assert worker.extended_leases == [(10, 17)]
    assert worker.completed_assets == [10]
    assert worker.failed_assets == []


def test_run_honors_max_jobs() -> None:
    processor = _FakeProcessor()
    worker = _WorkerHarness([_asset(1), _asset(2)], processor=processor)

    assert worker.run(max_jobs=1) == 1

    assert [asset.id for asset in processor.processed_assets] == [1]
    assert [asset.id for asset in worker.claimable_assets] == [2]


def test_lease_extension_failure_skips_indexing_and_completion() -> None:
    indexer = _FakeIndexer()
    worker = _WorkerHarness([_asset(20)], indexer=indexer, extend_result=False)

    assert worker.run_once() is True

    assert indexer.ingested_batches == []
    assert worker.completed_assets == []
    assert worker.failed_assets == []
    assert worker.extended_leases == [(20, 17)]


def test_processor_failure_logs_retry_outcome(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger=worker_module.__name__)
    processor = _FakeProcessor(error=RuntimeError("processor failed"))
    indexer = _FakeIndexer()
    worker = _WorkerHarness(
        [_asset(30)],
        processor=processor,
        indexer=indexer,
        fail_mark_result=AssetState.PENDING,
    )

    assert worker.run_once() is True

    assert indexer.ingested_batches == []
    assert worker.extended_leases == []
    assert worker.completed_assets == []
    assert worker.failed_assets == [(30, "processor failed")]
    assert "returning asset to queue" in caplog.text


def test_processor_failure_logs_permanent_failure_outcome(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.CRITICAL, logger=worker_module.__name__)
    processor = _FakeProcessor(error=RuntimeError("processor failed"))
    indexer = _FakeIndexer()
    worker = _WorkerHarness(
        [_asset(31)],
        processor=processor,
        indexer=indexer,
        fail_mark_result=AssetState.FAILED,
    )

    assert worker.run_once() is True

    assert indexer.ingested_batches == []
    assert worker.extended_leases == []
    assert worker.completed_assets == []
    assert worker.failed_assets == [(31, "processor failed")]
    assert "exhausted retries" in caplog.text


def test_processor_failure_logs_lost_ownership_outcome(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger=worker_module.__name__)
    processor = _FakeProcessor(error=RuntimeError("processor failed"))
    indexer = _FakeIndexer()
    worker = _WorkerHarness(
        [_asset(32)],
        processor=processor,
        indexer=indexer,
        fail_mark_result=None,
    )

    assert worker.run_once() is True

    assert indexer.ingested_batches == []
    assert worker.extended_leases == []
    assert worker.completed_assets == []
    assert worker.failed_assets == [(32, "processor failed")]
    assert "worker no longer owns lease" in caplog.text


def test_indexer_failure_marks_asset_failed() -> None:
    indexer = _FakeIndexer(error=RuntimeError("indexer failed"))
    worker = _WorkerHarness([_asset(40)], indexer=indexer)

    assert worker.run_once() is True

    assert worker.extended_leases == [(40, 17)]
    assert worker.completed_assets == []
    assert worker.failed_assets == [(40, "indexer failed")]


def test_completion_failure_after_indexing_logs_critical(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.CRITICAL, logger=worker_module.__name__)
    indexer = _FakeIndexer()
    worker = _WorkerHarness([_asset(50)], indexer=indexer, complete_result=False)

    assert worker.run_once() is True

    assert indexer.ingested_batches == [[]]
    assert worker.completed_assets == [50]
    assert "deterministic chunk IDs are required for safe retries" in caplog.text


def test_extend_asset_lease_forwards_extra_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int, int]] = []

    class _FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def begin(self):
            return self

    class _FakeDb:
        def get_session(self):
            return _FakeSession()

    class _FakeRepository:
        def __init__(self, session: _FakeSession) -> None:
            self.session = session

        def extend_asset_lease(
            self,
            worker_id: str,
            asset_id: int,
            extra_seconds: int = 60,
        ) -> bool:
            calls.append((worker_id, asset_id, extra_seconds))
            return True

    monkeypatch.setattr(worker_module, "SourceAssetRepository", _FakeRepository)
    worker = SourceIngestionWorkerService(
        db=_FakeDb(),
        settings=_settings(),
        processor=_FakeProcessor(),
        indexer=_FakeIndexer(),
        worker_id="worker-test",
    )

    assert worker._extend_asset_lease(60, extra_seconds=42) is True
    assert calls == [("worker-test", 60, 42)]
