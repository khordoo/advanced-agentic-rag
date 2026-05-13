import logging
from unittest.mock import patch

from rag_core.core.config import RetrievalSettings
import rag_core.retrieval.factory as factory_module
from rag_core.retrieval.factory import create_retriever


def test_factory_returns_qdrant_by_default() -> None:
    settings = RetrievalSettings(backend="qdrant", _env_file=None)
    with patch("rag_core.retrieval.factory.QdrantRetriever", return_value=object()) as mock_cls:
        result = create_retriever(settings=settings)
    assert result is not None
    mock_cls.assert_called_once_with(retrieval_settings=settings)


def test_factory_returns_postgres_when_configured() -> None:
    settings = RetrievalSettings(backend="postgres", _env_file=None)
    with patch("rag_core.retrieval.factory.PostgresRetriever", return_value=object()) as mock_cls:
        result = create_retriever(settings=settings)
    assert result is not None
    mock_cls.assert_called_once_with(settings=settings)


def test_factory_logs_selected_backend_and_retriever(caplog) -> None:
    class FakeRetriever:
        pass

    settings = RetrievalSettings(backend="qdrant", _env_file=None)
    caplog.set_level(logging.INFO, logger=factory_module.__name__)

    with patch("rag_core.retrieval.factory.QdrantRetriever", return_value=FakeRetriever()):
        create_retriever(settings=settings)

    assert "Retriever selected (backend=qdrant, retriever=FakeRetriever)" in caplog.text
