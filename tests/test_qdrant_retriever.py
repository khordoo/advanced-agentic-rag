from __future__ import annotations

from types import SimpleNamespace

import pytest
from qdrant_client import models

from rag_core.retrival.qdrant_retriever import QdrantRetriever
from rag_core.schemas import VectorDatabasePayloadKeys as PayloadKeys


def _retriever() -> QdrantRetriever:
    return object.__new__(QdrantRetriever)


def test_qdrant_retriever_filters_accept_asset_key_and_created_at() -> None:
    retriever = _retriever()

    qdrant_filter = retriever._build_filters(
        scopes=None,
        asset_key="asset-42",
        exact_filters={
            PayloadKeys.CREATED_AT.value: "2026-05-10T12:00:00+00:00",
        },
    )

    assert [condition.key for condition in qdrant_filter.must] == [
        PayloadKeys.ASSET_KEY.value,
        PayloadKeys.CREATED_AT.value,
    ]
    assert qdrant_filter.must[0].match.value == "asset-42"


def test_qdrant_retriever_rejects_file_name_filters() -> None:
    retriever = _retriever()

    with pytest.raises(KeyError, match=PayloadKeys.FILE_NAME.value):
        retriever._build_filters(
            scopes=None,
            exact_filters={PayloadKeys.FILE_NAME.value: "Asset 42.md"},
        )


def test_qdrant_retriever_documents_include_asset_key_and_created_at() -> None:
    retriever = _retriever()
    point = SimpleNamespace(
        id="parent-1",
        payload={
            PayloadKeys.TEXT.value: "parent text",
            PayloadKeys.ASSET_KEY.value: "asset-42",
            PayloadKeys.CREATED_AT.value: "2026-05-10T12:00:00+00:00",
            PayloadKeys.FILE_NAME.value: "Asset 42.md",
            "ignored": "metadata",
        },
    )

    doc = retriever._point_to_document(point)

    assert doc.page_content == "parent text"
    assert doc.metadata[PayloadKeys.ASSET_KEY.value] == "asset-42"
    assert doc.metadata[PayloadKeys.CREATED_AT.value] == "2026-05-10T12:00:00+00:00"
    assert "ignored" not in doc.metadata


def test_qdrant_retriever_still_groups_children_by_parent_id() -> None:
    retriever = _retriever()
    points = [
        SimpleNamespace(
            payload={PayloadKeys.PARENT_ID.value: "parent-1"},
            score=0.7,
        ),
        SimpleNamespace(
            payload={PayloadKeys.PARENT_ID.value: "parent-1"},
            score=0.9,
        ),
        SimpleNamespace(
            payload={PayloadKeys.PARENT_ID.value: "parent-2"},
            score=0.8,
        ),
    ]

    parent_scores = retriever._extract_unique_parents_and_filter(points, limit=2)

    assert parent_scores == [("parent-1", 0.9), ("parent-2", 0.8)]


class _FakeQdrantClient:
    def __init__(self) -> None:
        self.scroll_filter = None
        self.query_filters = []
        self.scroll_limit = None

    def scroll(self, **kwargs):
        self.scroll_filter = kwargs["scroll_filter"]
        self.scroll_limit = kwargs["limit"]
        point = SimpleNamespace(
            id="parent1",
            payload={
                PayloadKeys.TEXT.value: "parent text",
                PayloadKeys.ASSET_KEY.value: "asset-42",
            },
        )
        return [point], None

    def get_fastembed_vector_params(self):
        return ("dense",)

    def get_fastembed_sparse_vector_params(self):
        return ("sparse",)

    def _embed_documents(self, documents, embedding_model_name):
        return [("dense", [0.1])]

    def _sparse_embed_documents(self, documents, embedding_model_name):
        yield models.SparseVector(indices=[0], values=[0.1])

    @property
    def embedding_model_name(self):
        return "dense-model"

    @property
    def sparse_embedding_model_name(self):
        return "sparse-model"

    def query_points(self, **kwargs):
        self.query_filters = [prefetch.filter for prefetch in kwargs["prefetch"]]
        point = SimpleNamespace(
            id="child1",
            payload={PayloadKeys.PARENT_ID.value: "parent1"},
            score=0.8,
        )
        return SimpleNamespace(points=[point])

    def retrieve(self, **kwargs):
        return [
            SimpleNamespace(
                id="parent1",
                payload={
                    PayloadKeys.TEXT.value: "parent text",
                    PayloadKeys.ASSET_KEY.value: "asset-42",
                },
            )
        ]


def _configured_retriever(client: _FakeQdrantClient) -> QdrantRetriever:
    retriever = _retriever()
    retriever._client = client
    retriever._settings = SimpleNamespace(
        parent_collection_name="parents",
        chunk_collection_name="chunks",
    )
    retriever.retrieval_settings = SimpleNamespace(top_k=1, oversample_factor=1)
    return retriever


def test_exact_lookup_asset_key_builds_asset_key_filter() -> None:
    client = _FakeQdrantClient()
    retriever = _configured_retriever(client)

    docs = retriever.exact_lookup(asset_key="asset-42")

    assert docs[0].metadata[PayloadKeys.ASSET_KEY.value] == "asset-42"
    assert client.scroll_filter.must[0].key == PayloadKeys.ASSET_KEY.value
    assert client.scroll_filter.must[0].match.value == "asset-42"
    assert client.scroll_limit == 1


def test_search_asset_key_builds_asset_key_filter() -> None:
    client = _FakeQdrantClient()
    retriever = _configured_retriever(client)

    docs = retriever.search(query="revenue", asset_key="asset-42", rerank=False)

    assert docs[0].metadata[PayloadKeys.ASSET_KEY.value] == "asset-42"
    assert len(client.query_filters) == 2
    for qdrant_filter in client.query_filters:
        assert qdrant_filter.must[0].key == PayloadKeys.ASSET_KEY.value
        assert qdrant_filter.must[0].match.value == "asset-42"
