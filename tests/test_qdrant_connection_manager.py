from __future__ import annotations

import pytest

from rag_core.core.config import QdrantSettings
from rag_core.store.qdrant import connection_manager


def test_qdrant_connection_manager_passes_fastembed_cache_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class _FakeQdrantClient:
        def __init__(self, url: str) -> None:
            calls["url"] = url

        def get_collections(self) -> None:
            calls["ping"] = True

        def set_model(self, model_name: str, cache_dir: str | None = None) -> None:
            calls["dense"] = (model_name, cache_dir)

        def set_sparse_model(self, model_name: str, cache_dir: str | None = None) -> None:
            calls["sparse"] = (model_name, cache_dir)

    monkeypatch.setattr(connection_manager, "QdrantClient", _FakeQdrantClient)

    client = connection_manager.QdrantConnectionManager._connect_and_ping(
        QdrantSettings(
            url="http://qdrant:6333",
            embed_model="dense-model",
            sparse_model="sparse-model",
            fastembed_cache_dir="/app/.cache/fastembed",
            _env_file=None,
        )
    )

    assert isinstance(client, _FakeQdrantClient)
    assert calls["url"] == "http://qdrant:6333"
    assert calls["ping"] is True
    assert calls["dense"] == ("dense-model", "/app/.cache/fastembed")
    assert calls["sparse"] == ("sparse-model", "/app/.cache/fastembed")
