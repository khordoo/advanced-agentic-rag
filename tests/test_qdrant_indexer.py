from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.documents import Document
from qdrant_client import models

from rag_core.indexing.qdrant_indexer import QdrantIndexer
from rag_core.schemas import (
    CHILD_INDEX_SCHEMA,
    PARENT_INDEX_SCHEMA,
    ClaimedSourceAsset,
    ParentChunk,
    ParentMetadata,
    VectorDatabasePayloadKeys as PayloadKeys,
    _BASE_INDEX_SCHEMA,
)


def _asset(asset_key: str = "asset-42") -> ClaimedSourceAsset:
    return ClaimedSourceAsset(
        id=42,
        asset_key=asset_key,
        asset_name="asset-42.md",
        parent_name="folder",
        source_system="local_fs",
        source_uri="file:///asset-42.md",
        tags=["finance"],
        scopes=["public"],
        asset_metadata={},
        lease_version=1,
    )


def _block(
    asset_key: str = "asset-42",
    *,
    child_asset_key: str | None = None,
) -> ParentChunk:
    resolved_child_asset_key = asset_key if child_asset_key is None else child_asset_key
    return ParentChunk(
        id="parent-1",
        enriched_text="parent text",
        metadata=ParentMetadata(
            asset_key=asset_key,
            file_name="Asset 42.md",
            source="file:///asset-42.md",
            pages=[1],
            scopes=["public"],
            tags=["finance"],
        ),
        children=[
            Document(
                page_content="child text",
                metadata={
                    "asset_key": resolved_child_asset_key,
                    "file_name": "Asset 42.md",
                    "source": "file:///asset-42.md",
                    "pages": [1],
                    "parent_id": "parent-1",
                    "scopes": ["public"],
                    "tags": ["finance"],
                },
            )
        ],
    )


class _FakeQdrantClient:
    embedding_model_name = "dense-model"
    sparse_embedding_model_name = "sparse-model"

    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []
        self.deletes: list[tuple[str, models.FilterSelector]] = []
        self.upserts: list[tuple[str, list[models.PointStruct]]] = []

    def collection_exists(self, name: str) -> bool:
        return True

    def get_fastembed_vector_params(self):
        return {"dense": object()}

    def get_fastembed_sparse_vector_params(self):
        return {"sparse": object()}

    def _embed_documents(self, documents: list[str], embedding_model_name: str):
        return [("dense", [0.1, 0.2]) for _ in documents]

    def _sparse_embed_documents(
        self,
        documents: list[str],
        embedding_model_name: str,
    ):
        return [models.SparseVector(indices=[0], values=[0.5]) for _ in documents]

    def delete(
        self,
        *,
        collection_name: str,
        points_selector: models.FilterSelector,
        wait: bool,
    ) -> None:
        assert wait is True
        self.events.append(("delete", collection_name))
        self.deletes.append((collection_name, points_selector))

    def upsert(
        self,
        *,
        collection_name: str,
        points: list[models.PointStruct],
    ) -> None:
        self.events.append(("upsert", collection_name))
        self.upserts.append((collection_name, points))


def _indexer(client: _FakeQdrantClient) -> QdrantIndexer:
    return QdrantIndexer(
        client=client,
        settings=SimpleNamespace(
            chunk_collection_name="test_chunks",
            parent_collection_name="test_parents",
        ),
    )


def _filter_asset_key(selector: models.FilterSelector) -> str:
    condition = selector.filter.must[0]
    return condition.match.value


def test_qdrant_shared_payload_schema_indexes_asset_key_and_created_at() -> None:
    assert _BASE_INDEX_SCHEMA[PayloadKeys.ASSET_KEY.value] == (
        models.PayloadSchemaType.KEYWORD
    )
    assert _BASE_INDEX_SCHEMA[PayloadKeys.CREATED_AT.value] == (
        models.PayloadSchemaType.DATETIME
    )

    for schema in (PARENT_INDEX_SCHEMA, CHILD_INDEX_SCHEMA):
        assert schema[PayloadKeys.ASSET_KEY.value] == models.PayloadSchemaType.KEYWORD
        assert schema[PayloadKeys.CREATED_AT.value] == models.PayloadSchemaType.DATETIME


def test_index_asset_deletes_existing_points_before_upserting_fresh_points() -> None:
    client = _FakeQdrantClient()
    indexer = _indexer(client)

    indexer.index_asset(_asset(), [_block()])

    assert client.events == [
        ("delete", "test_chunks"),
        ("delete", "test_parents"),
        ("upsert", "test_chunks"),
        ("upsert", "test_parents"),
    ]
    assert [collection for collection, _ in client.deletes] == [
        "test_chunks",
        "test_parents",
    ]
    assert [_filter_asset_key(selector) for _, selector in client.deletes] == [
        "asset-42",
        "asset-42",
    ]

    _, child_points = client.upserts[0]
    _, parent_points = client.upserts[1]
    assert len(child_points) == 1
    assert len(parent_points) == 1
    assert PayloadKeys.CREATED_AT.value in child_points[0].payload
    assert PayloadKeys.CREATED_AT.value in parent_points[0].payload
    child_created_at = child_points[0].payload[PayloadKeys.CREATED_AT.value]
    parent_created_at = parent_points[0].payload[PayloadKeys.CREATED_AT.value]

    assert child_created_at == parent_created_at
    assert child_points[0].payload[PayloadKeys.ASSET_KEY.value] == "asset-42"
    assert parent_points[0].payload[PayloadKeys.ASSET_KEY.value] == "asset-42"


def test_index_asset_empty_blocks_deletes_without_upsert() -> None:
    client = _FakeQdrantClient()
    indexer = _indexer(client)

    indexer.index_asset(_asset(), [])

    assert client.events == [
        ("delete", "test_chunks"),
        ("delete", "test_parents"),
    ]
    assert client.upserts == []


def test_index_asset_rejects_parent_asset_key_mismatch_before_delete() -> None:
    client = _FakeQdrantClient()
    indexer = _indexer(client)

    with pytest.raises(ValueError, match="claimed asset"):
        indexer.index_asset(_asset("asset-42"), [_block("other-asset")])

    assert client.events == []


def test_index_asset_rejects_child_asset_key_mismatch_before_delete() -> None:
    client = _FakeQdrantClient()
    indexer = _indexer(client)

    with pytest.raises(ValueError, match="claimed asset"):
        indexer.index_asset(
            _asset("asset-42"),
            [_block("asset-42", child_asset_key="other-asset")],
        )

    assert client.events == []

