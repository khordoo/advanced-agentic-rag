import pytest
from pydantic import ValidationError

from rag_core.core.config import RetrievalSettings


def test_qdrant_rrf_k_unset_defaults_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RETRIVAL_QDRANT_RRF_K", raising=False)
    settings = RetrievalSettings(_env_file=None)
    assert settings.qdrant_rrf_k is None


def test_qdrant_rrf_k_empty_string_coerces_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETRIVAL_QDRANT_RRF_K", "")
    settings = RetrievalSettings(_env_file=None)
    assert settings.qdrant_rrf_k is None


def test_qdrant_rrf_k_whitespace_coerces_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETRIVAL_QDRANT_RRF_K", "   ")
    settings = RetrievalSettings(_env_file=None)
    assert settings.qdrant_rrf_k is None


def test_qdrant_rrf_k_null_like_strings_coerce_to_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for value in ["none", "NONE", "null", "NULL"]:
        monkeypatch.setenv("RETRIVAL_QDRANT_RRF_K", value)
        settings = RetrievalSettings(_env_file=None)
        assert settings.qdrant_rrf_k is None


def test_qdrant_rrf_k_valid_integer_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETRIVAL_QDRANT_RRF_K", "60")
    settings = RetrievalSettings(_env_file=None)
    assert settings.qdrant_rrf_k == 60


def test_qdrant_rrf_k_invalid_string_still_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETRIVAL_QDRANT_RRF_K", "abc")
    with pytest.raises(ValidationError, match="qdrant_rrf_k"):
        RetrievalSettings(_env_file=None)
