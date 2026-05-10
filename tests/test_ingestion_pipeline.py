"""Tests for canonical ingestion pipeline orchestration and the legacy shim.

The original IngestionPipeline has been deprecated in favour of
rag_core.ingestion.pipeline. These tests verify:

1. The shim still instantiates without error (backward compat).
2. The deprecation warning is raised on construction.
3. The canonical discovery, ingestion, and e2e entrypoints compose correctly.
4. The config loading defaults still work.

Tests for internal methods (_iter_files, _get_tags_for_file, etc.) have been
removed because those methods no longer exist on the shim.  Equivalent logic
is now covered by the SourceDiscoveryService and metadata extraction tests.
"""

from __future__ import annotations

from types import SimpleNamespace
import warnings
from pathlib import Path

import pytest

from rag_core.core.config import IngestionSettings, load_config
from rag_core.ingestion import pipeline as ingestion_pipeline
from rag_core.ingestion.services.discovery import DiscoveryRunMetrics
from rag_core.pipelines.ingestion import IngestionPipeline


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


def test_ingestion_pipeline_emits_deprecation_warning():
    """Constructing the legacy pipeline must warn about deprecation."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = IngestionPipeline(config={}, overwrite=False)
        deprecation_warnings = [
            x for x in w if issubclass(x.category, DeprecationWarning)
        ]
        assert len(deprecation_warnings) >= 1
        assert "deprecated" in str(deprecation_warnings[0].message).lower()


def test_ingestion_pipeline_shim_has_run_method():
    """The shim exposes a .run() method for backward compat."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        pipeline = IngestionPipeline(config={}, overwrite=False)
    assert hasattr(pipeline, "run")
    assert callable(pipeline.run)


def test_legacy_ingestion_pipeline_defaults_to_no_overwrite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []

    def _run_ingestion_pipeline(*, overwrite: bool) -> int:
        calls.append(overwrite)
        return 12

    monkeypatch.setattr(
        ingestion_pipeline,
        "run_ingestion_pipeline",
        _run_ingestion_pipeline,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        pipeline = IngestionPipeline(config={})

    assert pipeline.run() == 12
    assert calls == [False]


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

    assert ingestion_pipeline.run_ingestion_pipeline(overwrite=True) == 1
    assert worker.indexer.reset_count == 1


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
        lambda: expected,
    )

    assert ingestion_pipeline.run_pipeline() is expected


def test_legacy_ingestion_pipeline_run_delegates_to_canonical_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []

    def _run_ingestion_pipeline(*, overwrite: bool) -> int:
        calls.append(overwrite)
        return 12

    monkeypatch.setattr(
        ingestion_pipeline,
        "run_ingestion_pipeline",
        _run_ingestion_pipeline,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        pipeline = IngestionPipeline(config={}, overwrite=True)

    assert pipeline.run() == 12
    assert calls == [True]


@pytest.mark.skip(
    reason=(
        "Legacy IngestionPipeline._iter_files removed during deprecation. "
        "File iteration is now handled by SourceDiscoveryService + FileScanner."
    )
)
def test_iter_files_accepts_pdf_and_md_with_ignore_lists():
    pass


@pytest.mark.skip(
    reason=(
        "Legacy IngestionPipeline._get_tags_for_file removed during deprecation. "
        "Tag extraction is now handled by rag_core.ingestion.metadata."
    )
)
def test_folder_name_is_used_as_tag():
    pass


@pytest.mark.skip(
    reason=(
        "Legacy IngestionPipeline.run() now delegates to rag_core.ingestion.pipeline. "
        "E2E ingestion tests should target the canonical pipeline entrypoint."
    )
)
def test_run_skips_ignored_files_and_processes_pdf_and_md():
    pass
