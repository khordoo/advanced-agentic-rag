from unittest.mock import patch

import pytest

from rag_core.core.config import RerankerSettings
from rag_core.reranking.fastembed_reranker import FastEmbedReranker
from rag_core.reranking.factory import get_reranker


def test_get_reranker_defaults_to_fastembed() -> None:
    reranker = get_reranker(RerankerSettings(_env_file=None))

    assert isinstance(reranker, FastEmbedReranker)


def test_get_reranker_passes_settings_to_fastembed() -> None:
    settings = RerankerSettings(
        provider="FASTEMBED",
        model_name="BAAI/bge-reranker-base",
        top_k=7,
        _env_file=None,
    )

    with patch("rag_core.reranking.factory.FastEmbedReranker") as MockReranker:
        get_reranker(settings)

    MockReranker.assert_called_once_with(settings=settings)


def test_get_reranker_rejects_unknown_provider() -> None:
    settings = RerankerSettings(provider="cohere", _env_file=None)

    with pytest.raises(ValueError, match="Unsupported reranker provider: cohere"):
        get_reranker(settings)
