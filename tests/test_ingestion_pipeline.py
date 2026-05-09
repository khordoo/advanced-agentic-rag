"""Tests for the legacy IngestionPipeline deprecation shim.

The original IngestionPipeline has been deprecated in favour of
SourceIngestionWorkerService. These tests verify:

1. The shim still instantiates without error (backward compat).
2. The deprecation warning is raised on construction.
3. The config loading defaults still work.

Tests for internal methods (_iter_files, _get_tags_for_file, etc.) have been
removed because those methods no longer exist on the shim.  Equivalent logic
is now covered by the SourceDiscoveryService and metadata extraction tests.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from rag_core.core.config import IngestionSettings, load_config
from rag_core.pipelines.ingestion import IngestionPipeline


def test_load_config_defaults_docs_dir_to_source_documents(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("DOCS_DIR", raising=False)
    config = load_config(env_path=tmp_path / ".env.does.not.exist")
    assert config["DOCS_DIR"] == "data/source_documents"


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
        "Legacy IngestionPipeline.run() now delegates to SourceIngestionWorkerService. "
        "E2E ingestion tests should target the worker service directly."
    )
)
def test_run_skips_ignored_files_and_processes_pdf_and_md():
    pass
