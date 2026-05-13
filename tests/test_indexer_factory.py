"""Tests for the indexer factory and DI configuration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from rag_core.indexing.base import BaseIndexer
from rag_core.indexing.factory import IndexerBackend, create_indexer


@pytest.fixture(autouse=True)
def clear_settings_caches(monkeypatch):
    import rag_core.core.config

    monkeypatch.setenv("EMBEDDING_PROVIDER", "fastembed")
    rag_core.core.config.get_embedding_settings.cache_clear()
    rag_core.core.config.get_ingestion_settings.cache_clear()
    yield
    rag_core.core.config.get_embedding_settings.cache_clear()
    rag_core.core.config.get_ingestion_settings.cache_clear()


class TestIndexerBackendEnum:
    """IndexerBackend enum covers expected values."""

    def test_qdrant_value(self):
        assert IndexerBackend.QDRANT.value == "qdrant"

    def test_postgres_value(self):
        assert IndexerBackend.POSTGRES.value == "postgres"

    def test_string_coercion(self):
        assert IndexerBackend("qdrant") == IndexerBackend.QDRANT
        assert IndexerBackend("postgres") == IndexerBackend.POSTGRES

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError):
            IndexerBackend("redis")

    def test_old_pgvector_backend_raises(self):
        with pytest.raises(ValueError):
            IndexerBackend("pgvector")


class TestCreateIndexer:
    """Factory function returns correct indexer types."""

    def test_qdrant_backend_returns_base_indexer(self, monkeypatch):
        """When backend='qdrant', factory lazy-imports and returns QdrantIndexer."""
        monkeypatch.setenv("EMBEDDING_PROVIDER", "fastembed")
        mock_instance = MagicMock(spec=BaseIndexer)
        with patch(
            "rag_core.indexing.qdrant_indexer.QdrantIndexer",
            return_value=mock_instance,
        ) as mock_cls:
            result = create_indexer(backend="qdrant")
            assert result is mock_instance
            mock_cls.assert_called_once()

    def test_postgres_backend_returns_base_indexer(self):
        """When backend='postgres', factory lazy-imports and returns PostgresIndexer."""
        mock_instance = MagicMock(spec=BaseIndexer)
        with patch(
            "rag_core.indexing.postgres_indexer.PostgresIndexer",
            return_value=mock_instance,
        ) as mock_cls:
            result = create_indexer(backend="postgres")
            assert result is mock_instance
            mock_cls.assert_called_once()

    def test_invalid_backend_string_raises(self):
        """Unknown backend strings raise ValueError via enum coercion."""
        with pytest.raises(ValueError):
            create_indexer(backend="redis")

    def test_old_pgvector_backend_string_raises(self):
        """The old pgvector public backend value is not supported."""
        with pytest.raises(ValueError):
            create_indexer(backend="pgvector")

    def test_enum_member_accepted(self):
        """Factory accepts IndexerBackend enum members directly."""
        mock_instance = MagicMock(spec=BaseIndexer)
        with patch(
            "rag_core.indexing.postgres_indexer.PostgresIndexer",
            return_value=mock_instance,
        ):
            result = create_indexer(backend=IndexerBackend.POSTGRES)
            assert result is mock_instance

    def test_default_reads_from_settings(self, monkeypatch):
        """When backend=None, factory reads from IngestionSettings."""
        monkeypatch.setenv("EMBEDDING_PROVIDER", "fastembed")
        mock_settings = MagicMock()
        mock_settings.indexer_backend = "qdrant"

        mock_instance = MagicMock(spec=BaseIndexer)
        with (
            patch(
                "rag_core.core.config.get_ingestion_settings",
                return_value=mock_settings,
            ),
            patch(
                "rag_core.indexing.qdrant_indexer.QdrantIndexer",
                return_value=mock_instance,
            ),
        ):
            result = create_indexer(backend=None)
            assert result is mock_instance


class TestIngestionSettingsIndexerFields:
    """Verify the new config fields have correct defaults."""

    def test_default_indexer_backend(self, monkeypatch):
        monkeypatch.delenv("INGESTION_INDEXER_BACKEND", raising=False)
        from rag_core.core.config import IngestionSettings

        settings = IngestionSettings(_env_file=None)
        assert settings.indexer_backend == "qdrant"

    def test_custom_indexer_backend(self, monkeypatch):
        monkeypatch.setenv("INGESTION_INDEXER_BACKEND", "postgres")
        from rag_core.core.config import IngestionSettings

        settings = IngestionSettings(_env_file=None)
        assert settings.indexer_backend == "postgres"

    def test_old_pgvector_indexer_backend_rejected(self, monkeypatch):
        monkeypatch.setenv("INGESTION_INDEXER_BACKEND", "pgvector")
        from rag_core.core.config import IngestionSettings

        with pytest.raises(ValidationError):
            IngestionSettings(_env_file=None)
