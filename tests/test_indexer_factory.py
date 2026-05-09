"""Tests for the indexer factory and DI configuration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rag_core.indexing.base import BaseIndexer
from rag_core.indexing.factory import IndexerBackend, create_indexer


class TestIndexerBackendEnum:
    """IndexerBackend enum covers expected values."""

    def test_qdrant_value(self):
        assert IndexerBackend.QDRANT.value == "qdrant"

    def test_pgvector_value(self):
        assert IndexerBackend.PGVECTOR.value == "pgvector"

    def test_string_coercion(self):
        assert IndexerBackend("qdrant") == IndexerBackend.QDRANT
        assert IndexerBackend("pgvector") == IndexerBackend.PGVECTOR

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError):
            IndexerBackend("redis")


class TestCreateIndexer:
    """Factory function returns correct indexer types."""

    def test_qdrant_backend_returns_base_indexer(self):
        """When backend='qdrant', factory lazy-imports and returns QdrantIndexer."""
        mock_instance = MagicMock(spec=BaseIndexer)
        with patch(
            "rag_core.indexing.qdrant_indexer.QdrantIndexer",
            return_value=mock_instance,
        ) as mock_cls:
            result = create_indexer(backend="qdrant")
            assert result is mock_instance
            mock_cls.assert_called_once()

    def test_pgvector_backend_returns_base_indexer(self):
        """When backend='pgvector', factory lazy-imports and returns PgVectorIndexer."""
        mock_instance = MagicMock(spec=BaseIndexer)
        with patch(
            "rag_core.indexing.pgvector_indexer.PgVectorIndexer",
            return_value=mock_instance,
        ) as mock_cls:
            result = create_indexer(backend="pgvector")
            assert result is mock_instance
            mock_cls.assert_called_once()

    def test_invalid_backend_string_raises(self):
        """Unknown backend strings raise ValueError via enum coercion."""
        with pytest.raises(ValueError):
            create_indexer(backend="redis")

    def test_enum_member_accepted(self):
        """Factory accepts IndexerBackend enum members directly."""
        mock_instance = MagicMock(spec=BaseIndexer)
        with patch(
            "rag_core.indexing.pgvector_indexer.PgVectorIndexer",
            return_value=mock_instance,
        ):
            result = create_indexer(backend=IndexerBackend.PGVECTOR)
            assert result is mock_instance

    def test_default_reads_from_settings(self):
        """When backend=None, factory reads from IngestionSettings."""
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

    def test_default_embedding_dimension(self, monkeypatch):
        monkeypatch.delenv("INGESTION_EMBEDDING_DIMENSION", raising=False)
        from rag_core.core.config import IngestionSettings

        settings = IngestionSettings(_env_file=None)
        assert settings.embedding_dimension == 384

    def test_custom_indexer_backend(self, monkeypatch):
        monkeypatch.setenv("INGESTION_INDEXER_BACKEND", "pgvector")
        from rag_core.core.config import IngestionSettings

        settings = IngestionSettings()
        assert settings.indexer_backend == "pgvector"

    def test_custom_embedding_dimension(self, monkeypatch):
        monkeypatch.setenv("INGESTION_EMBEDDING_DIMENSION", "1536")
        from rag_core.core.config import IngestionSettings

        settings = IngestionSettings()
        assert settings.embedding_dimension == 1536
