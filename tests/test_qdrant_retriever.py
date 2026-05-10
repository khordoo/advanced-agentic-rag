from __future__ import annotations

from types import SimpleNamespace

from rag_core.retrival.qdrant_retriever import QdrantRetriever
from rag_core.schemas import VectorDatabasePayloadKeys as PayloadKeys


def _retriever() -> QdrantRetriever:
    return object.__new__(QdrantRetriever)


def test_qdrant_retriever_filters_accept_asset_key_and_created_at() -> None:
    retriever = _retriever()

    qdrant_filter = retriever._build_filters(
        scopes=None,
        exact_filters={
            PayloadKeys.ASSET_KEY.value: "asset-42",
            PayloadKeys.CREATED_AT.value: "2026-05-10T12:00:00+00:00",
        },
    )

    assert [condition.key for condition in qdrant_filter.must] == [
        PayloadKeys.ASSET_KEY.value,
        PayloadKeys.CREATED_AT.value,
    ]


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
