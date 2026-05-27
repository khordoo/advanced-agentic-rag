from unittest.mock import patch

from langchain_core.documents import Document
import pytest

from rag_core.core.config import RerankerSettings
from rag_core.reranking.fastembed_reranker import FastEmbedReranker


def test_reranker_settings_use_fastembed_base_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FASTEMBED_CACHE_DIR", "/app/.cache/fastembed")

    settings = RerankerSettings(_env_file=None)

    assert settings.provider == "fastembed"
    assert settings.model_name == "BAAI/bge-reranker-base"
    assert settings.fastembed_cache_dir == "/app/.cache/fastembed"


def test_fastembed_reranker_sorts_correctly() -> None:
    docs = [
        Document(page_content="doc0", metadata={}),
        Document(page_content="doc1", metadata={}),
        Document(page_content="doc2", metadata={}),
    ]

    with patch("rag_core.reranking.fastembed_reranker.TextCrossEncoder") as MockEncoder:
        mock_instance = MockEncoder.return_value
        mock_instance.rerank.return_value = [0.1, 0.9, 0.5]

        reranker = FastEmbedReranker(
            settings=RerankerSettings(
                top_k=2,
                model_name="BAAI/bge-reranker-base",
                fastembed_cache_dir="/app/.cache/fastembed",
                _env_file=None,
            )
        )
        results = reranker.rerank("query", docs)

        MockEncoder.assert_called_once_with(
            model_name="BAAI/bge-reranker-base",
            cache_dir="/app/.cache/fastembed",
        )
        assert len(results) == 2
        assert results[0].page_content == "doc1"
        assert results[0].metadata["rerank_score"] == 0.9
        assert results[1].page_content == "doc2"
        assert results[1].metadata["rerank_score"] == 0.5
