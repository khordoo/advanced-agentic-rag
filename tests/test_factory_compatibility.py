"""Compatibility guard tests for indexer and embedding backends."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from rag_core.core.config import IngestionSettings, QdrantSettings
from rag_core.indexing.base import BaseIndexer
from rag_core.indexing.factory import (
    IndexerBackend,
    create_indexer,
    validate_indexing_runtime_compatibility,
)
from rag_core.ingestion import pipeline as ingestion_pipeline
from rag_core.store.postgres.models.documents import DocumentChunk


@pytest.fixture(autouse=True)
def clear_settings_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    import rag_core.core.config

    monkeypatch.setenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
    monkeypatch.setenv("QDRANT_SPARSE_MODEL", "Qdrant/bm42-all-minilm-l6-v2-attentions")
    rag_core.core.config.get_embedding_settings.cache_clear()
    rag_core.core.config.get_ingestion_settings.cache_clear()
    rag_core.core.config.get_qdrant_settings.cache_clear()
    yield
    rag_core.core.config.get_embedding_settings.cache_clear()
    rag_core.core.config.get_ingestion_settings.cache_clear()
    rag_core.core.config.get_qdrant_settings.cache_clear()


class _FakeScalarResult:
    def __init__(self, values: list[int | None]) -> None:
        self._values = values

    def all(self) -> list[int | None]:
        return self._values


class _FakeExecuteResult:
    def __init__(self, values: list[int | None]) -> None:
        self._values = values

    def scalars(self) -> _FakeScalarResult:
        return _FakeScalarResult(self._values)


class _FakeSession:
    def __init__(self, dimensions: list[int | None]) -> None:
        self._dimensions = dimensions
        self.executed_sql = ""

    def __enter__(self) -> "_FakeSession":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, stmt: object) -> _FakeExecuteResult:
        self.executed_sql = str(stmt)
        return _FakeExecuteResult(self._dimensions)


class _FakeDb:
    def __init__(self, dimensions: list[int | None]) -> None:
        self.session = _FakeSession(dimensions)

    def get_session(self) -> _FakeSession:
        return self.session


def test_create_indexer_qdrant_rejects_openai_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")

    with pytest.raises(ValueError, match="Qdrant currently requires") as exc_info:
        create_indexer("qdrant")

    message = str(exc_info.value)
    assert "EMBEDDING_PROVIDER=fastembed" in message
    assert "EMBEDDING_MODEL_NAME=BAAI/bge-small-en-v1.5" in message
    assert "QDRANT_SPARSE_MODEL=Qdrant/bm42-all-minilm-l6-v2-attentions" in message


def test_qdrant_settings_reject_empty_dense_model() -> None:
    with pytest.raises(ValidationError, match="Qdrant model names must not be empty"):
        QdrantSettings(embed_model=" ", _env_file=None)


def test_qdrant_settings_reads_shared_embedding_model_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_MODEL_NAME", "custom-dense-model")
    monkeypatch.setenv("FASTEMBED_CACHE_DIR", "/app/.cache/fastembed")
    monkeypatch.setenv("QDRANT_EMBED_MODEL", "legacy-qdrant-model")

    settings = QdrantSettings(_env_file=None)

    assert settings.embed_model == "custom-dense-model"
    assert settings.fastembed_cache_dir == "/app/.cache/fastembed"


def test_qdrant_settings_reject_empty_sparse_model() -> None:
    with pytest.raises(ValidationError, match="Qdrant model names must not be empty"):
        QdrantSettings(sparse_model=" ", _env_file=None)


def test_ingestion_settings_overwrite_defaults_to_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INGESTION_OVERWRITE", raising=False)

    settings = IngestionSettings(_env_file=None)

    assert settings.overwrite is False


def test_ingestion_settings_overwrite_reads_runtime_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INGESTION_OVERWRITE", "true")

    settings = IngestionSettings(_env_file=None)

    assert settings.overwrite is True


def test_create_indexer_qdrant_allows_fastembed_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fastembed")

    mock_instance = MagicMock(spec=BaseIndexer)
    with patch(
        "rag_core.indexing.qdrant_indexer.QdrantIndexer",
        return_value=mock_instance,
    ) as mock_cls:
        assert create_indexer("qdrant") is mock_instance

    mock_cls.assert_called_once()


def test_create_indexer_postgres_allows_fastembed_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fastembed")

    mock_instance = MagicMock(spec=BaseIndexer)
    with patch(
        "rag_core.indexing.postgres_indexer.PostgresIndexer",
        return_value=mock_instance,
    ) as mock_cls:
        assert create_indexer("postgres") is mock_instance

    mock_cls.assert_called_once()


def test_create_indexer_postgres_openai_fails_from_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")

    with (
        patch("rag_core.indexing.postgres_indexer.get_database", return_value=object()),
        pytest.raises(NotImplementedError, match="OpenAI embeddings"),
    ):
        create_indexer("postgres")


def test_postgres_runtime_preflight_allows_empty_vector_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_DIMENSION", "384")
    db = _FakeDb([])

    result = validate_indexing_runtime_compatibility("postgres", db=db)

    assert result.backend == IndexerBackend.POSTGRES
    assert result.configured_dimension == 384
    assert result.existing_dimensions == ()
    assert result.overwrite_rebuild is False
    assert f"FROM {DocumentChunk.__tablename__}" in db.session.executed_sql
    assert "vector_dims(embedding)" in db.session.executed_sql
    assert "WHERE embedding IS NOT NULL" in db.session.executed_sql


def test_postgres_runtime_preflight_allows_matching_existing_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_DIMENSION", "384")

    result = validate_indexing_runtime_compatibility("postgres", db=_FakeDb([384]))

    assert result.backend == IndexerBackend.POSTGRES
    assert result.configured_dimension == 384
    assert result.existing_dimensions == (384,)
    assert result.overwrite_rebuild is False


def test_postgres_runtime_preflight_rejects_existing_dimension_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_DIMENSION", "384")

    with pytest.raises(ValueError, match="Dimension mismatch detected") as exc_info:
        validate_indexing_runtime_compatibility("postgres", db=_FakeDb([768]))

    message = str(exc_info.value)
    assert "INGESTION_OVERWRITE=true" in message
    assert "overwrite=True" in message
    assert "delete all existing vectors" in message


def test_postgres_runtime_preflight_rejects_mixed_existing_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_DIMENSION", "384")

    with pytest.raises(ValueError, match="Dimension mismatch detected"):
        validate_indexing_runtime_compatibility("postgres", db=_FakeDb([384, 768]))


def test_postgres_runtime_preflight_allows_mismatch_with_overwrite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDING_DIMENSION", "384")

    result = validate_indexing_runtime_compatibility(
        "postgres",
        overwrite=True,
        db=_FakeDb([768]),
    )

    assert result.backend == IndexerBackend.POSTGRES
    assert result.configured_dimension == 384
    assert result.existing_dimensions == (768,)
    assert result.overwrite_rebuild is True


def test_run_ingestion_pipeline_validates_postgres_before_worker_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INGESTION_INDEXER_BACKEND", "postgres")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "384")

    def _worker_should_not_be_built():
        raise AssertionError("worker should not be constructed")

    monkeypatch.setattr(
        "rag_core.indexing.factory._fetch_existing_postgres_vector_dimensions",
        lambda *, db=None: {768},
    )
    monkeypatch.setattr(
        ingestion_pipeline.SourceIngestionWorkerService,
        "default_production",
        staticmethod(_worker_should_not_be_built),
    )

    with pytest.raises(ValueError, match="Dimension mismatch detected"):
        ingestion_pipeline.run_ingestion_pipeline()


def test_run_end_to_end_pipeline_validates_before_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INGESTION_INDEXER_BACKEND", "qdrant")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")

    def _discovery_should_not_run():
        raise AssertionError("discovery should not run")

    monkeypatch.setattr(
        ingestion_pipeline,
        "run_discovery_pipeline",
        _discovery_should_not_run,
    )

    with pytest.raises(ValueError, match="Qdrant currently requires"):
        ingestion_pipeline.run_end_to_end_pipeline()


def test_run_ingestion_pipeline_validates_before_worker_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INGESTION_INDEXER_BACKEND", "qdrant")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")

    def _worker_should_not_be_built():
        raise AssertionError("worker should not be constructed")

    monkeypatch.setattr(
        ingestion_pipeline.SourceIngestionWorkerService,
        "default_production",
        staticmethod(_worker_should_not_be_built),
    )

    with pytest.raises(ValueError, match="Qdrant currently requires"):
        ingestion_pipeline.run_ingestion_pipeline()
