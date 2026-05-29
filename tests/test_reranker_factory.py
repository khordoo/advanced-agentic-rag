from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document

from rag_core.core.config import RerankerSettings
from rag_core.reranking.bedrock_reranker import BedrockReranker
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


def test_get_reranker_returns_bedrock_reranker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
    settings = RerankerSettings(
        provider="bedrock",
        model_name="amazon.rerank-v1:0",
        top_k=3,
        _env_file=None,
    )

    with patch("rag_core.reranking.factory.BedrockReranker") as MockReranker:
        get_reranker(settings)

    MockReranker.assert_called_once_with(settings=settings)


def test_bedrock_reranker_builds_expected_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
    docs = [
        Document(page_content="doc0", metadata={}),
        Document(page_content="doc1", metadata={}),
        Document(page_content="doc2", metadata={}),
    ]

    mock_client = Mock()
    mock_client.rerank.return_value = {
        "results": [
            {"index": 1, "relevanceScore": 0.9},
            {"index": 2, "relevanceScore": 0.5},
        ]
    }

    with patch("rag_core.reranking.bedrock_reranker.boto3.client", return_value=mock_client) as mock_boto:
        reranker = BedrockReranker(
            settings=RerankerSettings(
                provider="bedrock",
                model_name="amazon.rerank-v1:0",
                top_k=2,
                _env_file=None,
            )
        )
        results = reranker.rerank("query", docs)

    mock_boto.assert_called_once_with("bedrock-agent-runtime", region_name="us-west-2")
    mock_client.rerank.assert_called_once()
    assert mock_client.rerank.call_args.kwargs == {
        "queries": [{"type": "TEXT", "textQuery": {"text": "query"}}],
        "sources": [
            {
                "type": "INLINE",
                "inlineDocumentSource": {
                    "type": "TEXT",
                    "textDocument": {"text": "doc0"},
                },
            },
            {
                "type": "INLINE",
                "inlineDocumentSource": {
                    "type": "TEXT",
                    "textDocument": {"text": "doc1"},
                },
            },
            {
                "type": "INLINE",
                "inlineDocumentSource": {
                    "type": "TEXT",
                    "textDocument": {"text": "doc2"},
                },
            },
        ],
        "rerankingConfiguration": {
            "type": "BEDROCK_RERANKING_MODEL",
            "bedrockRerankingConfiguration": {
                "modelConfiguration": {
                    "modelArn": "arn:aws:bedrock:us-west-2::foundation-model/amazon.rerank-v1:0"
                },
                "numberOfResults": 2,
            },
        },
    }
    assert len(results) == 2
    assert results[0].page_content == "doc1"
    assert results[0].metadata["rerank_score"] == 0.9
    assert results[1].page_content == "doc2"
    assert results[1].metadata["rerank_score"] == 0.5


def test_bedrock_reranker_accepts_full_model_arn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-west-2")
    mock_client = Mock()
    mock_client.rerank.return_value = {"results": []}

    with patch("rag_core.reranking.bedrock_reranker.boto3.client", return_value=mock_client):
        reranker = BedrockReranker(
            settings=RerankerSettings(
                provider="bedrock",
                model_name="arn:aws:bedrock:us-west-2::foundation-model/amazon.rerank-v1:0",
                _env_file=None,
            )
        )
        reranker.rerank("query", [Document(page_content="doc0", metadata={})])

    assert mock_client.rerank.call_args.kwargs["rerankingConfiguration"][
        "bedrockRerankingConfiguration"
    ]["modelConfiguration"]["modelArn"] == (
        "arn:aws:bedrock:us-west-2::foundation-model/amazon.rerank-v1:0"
    )
