from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_core.core.config import EmbeddingSettings
from rag_core.embeddings import (
    BaseEmbeddingModel,
    FastEmbedEmbeddingModel,
    OpenAIEmbeddingModel,
    create_embedding_model,
)


class _ArrayLike:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def tolist(self) -> list[float]:
        return self.values


def test_base_embedding_model_contract_with_fake() -> None:
    class _FakeEmbeddingModel(BaseEmbeddingModel):
        @property
        def dimension(self) -> int:
            return 2

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[float(len(text)), 1.0] for text in texts]

        def embed_query(self, text: str) -> list[float]:
            return self.embed_documents([text])[0]

    model = _FakeEmbeddingModel()

    assert model.dimension == 2
    assert model.embed_documents(["a", "abcd"]) == [[1.0, 1.0], [4.0, 1.0]]
    assert model.embed_query("abc") == [3.0, 1.0]


def test_fastembed_embedding_model_lazy_loads_and_embeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_models: list[str] = []

    class _FakeTextEmbedding:
        def __init__(self, model_name: str) -> None:
            loaded_models.append(model_name)

        def embed(self, texts: list[str]):
            assert texts == ["alpha", "beta"]
            return [_ArrayLike([1, 2]), [3.5, 4]]

    monkeypatch.setattr(
        "rag_core.embeddings.fastembed._get_fastembed_model",
        _FakeTextEmbedding,
    )

    model = FastEmbedEmbeddingModel(model_name="local-model", dimension=2)

    assert model.dimension == 2
    assert loaded_models == []
    assert model.embed_documents(["alpha", "beta"]) == [[1.0, 2.0], [3.5, 4.0]]
    assert loaded_models == ["local-model"]


def test_fastembed_embedding_model_embed_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeTextEmbedding:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def embed(self, texts: list[str]):
            assert texts == ["query text"]
            return [[0.1, 0.2, 0.3]]

    monkeypatch.setattr(
        "rag_core.embeddings.fastembed._get_fastembed_model",
        _FakeTextEmbedding,
    )

    model = FastEmbedEmbeddingModel(model_name="local-model", dimension=3)

    assert model.embed_query("query text") == [0.1, 0.2, 0.3]


def test_fastembed_embedding_model_rejects_dimension_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeTextEmbedding:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def embed(self, texts: list[str]):
            del texts
            return [[0.1]]

    monkeypatch.setattr(
        "rag_core.embeddings.fastembed._get_fastembed_model",
        _FakeTextEmbedding,
    )
    model = FastEmbedEmbeddingModel(model_name="local-model", dimension=2)

    with pytest.raises(ValueError, match="Embedding dimension mismatch"):
        model.embed_documents(["text"])


def test_create_embedding_model_returns_fastembed() -> None:
    settings = EmbeddingSettings(
        provider="fastembed",
        model_name="local-model",
        dimension=384,
        _env_file=None,
    )

    model = create_embedding_model(settings)

    assert isinstance(model, FastEmbedEmbeddingModel)
    assert model.model_name == "local-model"
    assert model.dimension == 384


def test_embedding_settings_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fastembed")
    monkeypatch.setenv("EMBEDDING_MODEL_NAME", "custom-model")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "768")

    settings = EmbeddingSettings(_env_file=None)

    assert settings.provider == "fastembed"
    assert settings.model_name == "custom-model"
    assert settings.dimension == 768


def test_create_embedding_model_returns_openai_placeholder() -> None:
    settings = EmbeddingSettings(
        provider="openai",
        model_name="text-embedding-3-small",
        dimension=1536,
        _env_file=None,
    )

    model = create_embedding_model(settings)

    assert isinstance(model, OpenAIEmbeddingModel)
    assert model.dimension == 1536
    with pytest.raises(NotImplementedError, match="OpenAI embeddings"):
        model.embed_query("query")


def test_embedding_settings_rejects_unknown_provider() -> None:
    with pytest.raises(ValidationError):
        EmbeddingSettings(provider="unknown", _env_file=None)
