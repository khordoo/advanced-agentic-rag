"""Tests for canonical ingestion pipeline orchestration."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from pathlib import Path

import pytest

from rag_core.core.config import IngestionSettings, load_config
from rag_core.ingestion import pipeline as ingestion_pipeline
from rag_core.ingestion import indexing_preflight
from rag_core.ingestion.services.discovery import DiscoveryRunMetrics


@pytest.fixture(autouse=True)
def clear_settings_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    import rag_core.core.config

    monkeypatch.setenv("EMBEDDING_PROVIDER", "fastembed")
    monkeypatch.setenv("INGESTION_INDEXER_BACKEND", "qdrant")
    monkeypatch.setenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
    monkeypatch.setenv("QDRANT_SPARSE_MODEL", "Qdrant/bm42-all-minilm-l6-v2-attentions")
    rag_core.core.config.get_embedding_settings.cache_clear()
    rag_core.core.config.get_ingestion_settings.cache_clear()
    rag_core.core.config.get_qdrant_settings.cache_clear()
    yield
    rag_core.core.config.get_embedding_settings.cache_clear()
    rag_core.core.config.get_ingestion_settings.cache_clear()
    rag_core.core.config.get_qdrant_settings.cache_clear()


def test_load_config_defaults_docs_dir_to_source_documents(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("DOCS_DIR", raising=False)
    config = load_config(env_path=tmp_path / ".env.does.not.exist")
    assert config["DOCS_DIR"] == "data/source_documents"


def test_ingestion_settings_default_max_jobs_defaults_to_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INGESTION_INDEXER_BACKEND", "qdrant")
    monkeypatch.delenv("INGESTION_DEFAULT_MAX_JOBS", raising=False)

    settings = IngestionSettings(_env_file=None)

    assert settings.default_max_jobs is None


def test_ingestion_settings_default_max_jobs_reads_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INGESTION_INDEXER_BACKEND", "qdrant")
    monkeypatch.setenv("INGESTION_DEFAULT_MAX_JOBS", "100")

    settings = IngestionSettings(_env_file=None)

    assert settings.default_max_jobs == 100


def test_run_discovery_pipeline_delegates_to_discovery_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = DiscoveryRunMetrics(discovered_total=2)

    monkeypatch.setattr(ingestion_pipeline, "run_discovery", lambda: metrics)

    assert ingestion_pipeline.run_discovery_pipeline() is metrics


def test_run_ingestion_pipeline_runs_worker_without_reset_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeIndexer:
        def __init__(self) -> None:
            self.reset_count = 0

        def reset_collections(self) -> None:
            self.reset_count += 1

    class _FakeWorker:
        def __init__(self) -> None:
            self.indexer = _FakeIndexer()
            self.settings = SimpleNamespace(default_max_jobs=100)
            self.max_jobs_seen: int | None = None

        def run(self, max_jobs: int | None = None) -> int:
            self.max_jobs_seen = max_jobs
            return 7

    worker = _FakeWorker()
    monkeypatch.setattr(
        ingestion_pipeline.SourceIngestionWorkerService,
        "default_production",
        staticmethod(lambda: worker),
    )

    processed = ingestion_pipeline.run_ingestion_pipeline(max_jobs=3)

    assert processed == 7
    assert worker.max_jobs_seen == 3
    assert worker.indexer.reset_count == 0


def test_run_ingestion_pipeline_uses_config_default_when_max_jobs_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeIndexer:
        def reset_collections(self) -> None:
            raise AssertionError("reset_collections should not be called")

    class _FakeWorker:
        def __init__(self) -> None:
            self.indexer = _FakeIndexer()
            self.settings = SimpleNamespace(default_max_jobs=11)
            self.max_jobs_seen: int | None = None

        def run(self, max_jobs: int | None = None) -> int:
            self.max_jobs_seen = max_jobs
            return 7

    worker = _FakeWorker()
    monkeypatch.setattr(
        ingestion_pipeline.SourceIngestionWorkerService,
        "default_production",
        staticmethod(lambda: worker),
    )

    assert ingestion_pipeline.run_ingestion_pipeline() == 7
    assert worker.max_jobs_seen == 11


def test_run_ingestion_pipeline_preserves_unlimited_when_config_default_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeIndexer:
        def reset_collections(self) -> None:
            raise AssertionError("reset_collections should not be called")

    class _FakeWorker:
        def __init__(self) -> None:
            self.indexer = _FakeIndexer()
            self.settings = SimpleNamespace(default_max_jobs=None)
            self.max_jobs_seen = "unset"

        def run(self, max_jobs: int | None = None) -> int:
            self.max_jobs_seen = max_jobs
            return 7

    worker = _FakeWorker()
    monkeypatch.setattr(
        ingestion_pipeline.SourceIngestionWorkerService,
        "default_production",
        staticmethod(lambda: worker),
    )

    assert ingestion_pipeline.run_ingestion_pipeline() == 7
    assert worker.max_jobs_seen is None


def test_run_ingestion_pipeline_preserves_zero_max_jobs_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeIndexer:
        def reset_collections(self) -> None:
            raise AssertionError("reset_collections should not be called")

    class _FakeWorker:
        def __init__(self) -> None:
            self.indexer = _FakeIndexer()
            self.settings = SimpleNamespace(default_max_jobs=11)
            self.max_jobs_seen: int | None = None

        def run(self, max_jobs: int | None = None) -> int:
            self.max_jobs_seen = max_jobs
            return 0

    worker = _FakeWorker()
    monkeypatch.setattr(
        ingestion_pipeline.SourceIngestionWorkerService,
        "default_production",
        staticmethod(lambda: worker),
    )

    assert ingestion_pipeline.run_ingestion_pipeline(max_jobs=0) == 0
    assert worker.max_jobs_seen == 0


def test_run_ingestion_pipeline_resets_when_overwrite_enabled(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _FakeIndexer:
        def __init__(self) -> None:
            self.reset_count = 0

        def reset_collections(self) -> None:
            self.reset_count += 1

    class _FakeWorker:
        def __init__(self) -> None:
            self.indexer = _FakeIndexer()
            self.settings = SimpleNamespace(default_max_jobs=None)

        def run(self, max_jobs: int | None = None) -> int:
            return 1

    worker = _FakeWorker()
    monkeypatch.setattr(
        ingestion_pipeline.SourceIngestionWorkerService,
        "default_production",
        staticmethod(lambda: worker),
    )

    caplog.set_level(logging.INFO, logger=indexing_preflight.__name__)
    assert ingestion_pipeline.run_ingestion_pipeline(overwrite=True) == 1
    assert worker.indexer.reset_count == 1
    assert "Existing vectors for this backend will be deleted" in caplog.text
    assert "Overwrite successful" not in caplog.text


def test_run_ingestion_pipeline_logs_postgres_mismatch_overwrite_rebuild(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("INGESTION_INDEXER_BACKEND", "postgres")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "1536")

    class _FakeIndexer:
        def __init__(self) -> None:
            self.reset_count = 0

        def reset_collections(self) -> None:
            self.reset_count += 1

    class _FakeWorker:
        def __init__(self) -> None:
            self.indexer = _FakeIndexer()
            self.settings = SimpleNamespace(default_max_jobs=None)

        def run(self, max_jobs: int | None = None) -> int:
            return 1

    worker = _FakeWorker()
    monkeypatch.setattr(
        "rag_core.indexing.factory._fetch_existing_postgres_vector_dimensions",
        lambda *, db=None: {384},
    )
    monkeypatch.setattr(
        ingestion_pipeline.SourceIngestionWorkerService,
        "default_production",
        staticmethod(lambda: worker),
    )

    caplog.set_level(logging.INFO, logger=indexing_preflight.__name__)
    assert ingestion_pipeline.run_ingestion_pipeline(overwrite=True) == 1

    assert worker.indexer.reset_count == 1
    assert (
        "Dimension mismatch detected (DB: 384, Config: 1536). Proceeding with "
        "index reset as requested."
    ) in caplog.text
    assert "Overwrite successful. Index rebuilt with dimension 1536." in caplog.text
    assert "run future ingestion with overwrite=False" in caplog.text


def test_run_end_to_end_pipeline_runs_discovery_then_ingestion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    metrics = DiscoveryRunMetrics(discovered_total=4)

    def _run_discovery_pipeline() -> DiscoveryRunMetrics:
        calls.append("discovery")
        return metrics

    def _run_ingestion_pipeline(*, overwrite: bool, max_jobs: int | None) -> int:
        calls.append(f"ingestion:{overwrite}:{max_jobs}")
        return 9

    monkeypatch.setattr(
        ingestion_pipeline,
        "run_discovery_pipeline",
        _run_discovery_pipeline,
    )
    monkeypatch.setattr(
        ingestion_pipeline,
        "run_ingestion_pipeline",
        _run_ingestion_pipeline,
    )

    result = ingestion_pipeline.run_end_to_end_pipeline(overwrite=True, max_jobs=5)

    assert calls == ["discovery", "ingestion:True:5"]
    assert result == ingestion_pipeline.EndToEndIngestionResult(
        discovery=metrics,
        processed_jobs=9,
        succeeded=True,
    )


def test_run_end_to_end_pipeline_skips_ingestion_when_discovery_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ingestion_calls = 0

    def _run_ingestion_pipeline(*, overwrite: bool, max_jobs: int | None) -> int:
        nonlocal ingestion_calls
        ingestion_calls += 1
        return 9

    monkeypatch.setattr(ingestion_pipeline, "run_discovery_pipeline", lambda: None)
    monkeypatch.setattr(
        ingestion_pipeline,
        "run_ingestion_pipeline",
        _run_ingestion_pipeline,
    )

    result = ingestion_pipeline.run_end_to_end_pipeline()

    assert ingestion_calls == 0
    assert result == ingestion_pipeline.EndToEndIngestionResult(
        discovery=None,
        processed_jobs=0,
        succeeded=False,
    )


def test_run_pipeline_delegates_to_end_to_end_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = ingestion_pipeline.EndToEndIngestionResult(
        discovery=DiscoveryRunMetrics(discovered_total=1),
        processed_jobs=2,
        succeeded=True,
    )

    monkeypatch.setattr(
        ingestion_pipeline,
        "run_end_to_end_pipeline",
        lambda *, overwrite: expected,
    )

    assert ingestion_pipeline.run_pipeline() is expected


def test_run_pipeline_reads_runtime_overwrite_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INGESTION_OVERWRITE", "true")
    calls: list[bool] = []
    expected = ingestion_pipeline.EndToEndIngestionResult(
        discovery=DiscoveryRunMetrics(discovered_total=1),
        processed_jobs=2,
        succeeded=True,
    )

    def _run_end_to_end_pipeline(*, overwrite: bool):
        calls.append(overwrite)
        return expected

    monkeypatch.setattr(
        ingestion_pipeline,
        "run_end_to_end_pipeline",
        _run_end_to_end_pipeline,
    )

    assert ingestion_pipeline.run_pipeline() is expected
    assert calls == [True]
