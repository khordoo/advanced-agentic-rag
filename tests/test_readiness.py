from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy.exc import SQLAlchemyError

from rag_core.core import readiness
from rag_core.core.config import RetrievalSettings
from rag_core.core.readiness import ApplicationReadinessChecker, SetupRequiredError


class _Inspector:
    def __init__(self, existing_tables: set[str]) -> None:
        self._existing_tables = existing_tables

    def has_table(self, table_name: str) -> bool:
        return table_name in self._existing_tables


class _Session:
    def __init__(self, scalar_value: bool) -> None:
        self._scalar_value = scalar_value

    def scalar(self, _statement):
        return self._scalar_value


class _SessionContext:
    def __init__(self, session: _Session) -> None:
        self._session = session

    def __enter__(self) -> _Session:
        return self._session

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _Database:
    engine = object()

    def __init__(self, scalar_value: bool = True) -> None:
        self._scalar_value = scalar_value

    def get_session(self):
        return _SessionContext(_Session(self._scalar_value))


def _postgres_checker(db: _Database) -> ApplicationReadinessChecker:
    return ApplicationReadinessChecker(
        db=db,
        retrieval_settings=RetrievalSettings(backend="postgres"),
    )


def test_readiness_reports_missing_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(readiness, "inspect", lambda _engine: _Inspector({"source_assets"}))

    checker = _postgres_checker(_Database())

    with pytest.raises(SetupRequiredError) as exc_info:
        checker.ensure_ready()

    assert exc_info.value.reason == "schema_not_migrated"
    assert "make ingest" in exc_info.value.message


def test_readiness_reports_database_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(_engine):
        raise SQLAlchemyError("connection refused")

    monkeypatch.setattr(readiness, "inspect", _raise)

    checker = _postgres_checker(_Database())

    with pytest.raises(SetupRequiredError) as exc_info:
        checker.ensure_ready()

    assert exc_info.value.reason == "database_unavailable"
    assert "make up" in exc_info.value.message


def test_readiness_reports_empty_postgres_knowledge_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        readiness,
        "inspect",
        lambda _engine: _Inspector(set(readiness.REQUIRED_TABLES)),
    )

    checker = _postgres_checker(_Database(scalar_value=False))

    with pytest.raises(SetupRequiredError) as exc_info:
        checker.ensure_ready()

    assert exc_info.value.reason == "knowledge_base_empty"
    assert "make ingest" in exc_info.value.message


def test_readiness_passes_when_postgres_has_indexed_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        readiness,
        "inspect",
        lambda _engine: _Inspector(set(readiness.REQUIRED_TABLES)),
    )

    checker = _postgres_checker(_Database(scalar_value=True))

    checker.ensure_ready()


def test_readiness_reports_empty_qdrant_knowledge_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        readiness,
        "inspect",
        lambda _engine: _Inspector(set(readiness.REQUIRED_TABLES)),
    )
    monkeypatch.setattr(
        readiness,
        "get_qdrant_settings",
        lambda: SimpleNamespace(parent_collection_name="parents"),
    )

    class _QdrantClient:
        def scroll(self, **kwargs):
            assert kwargs["collection_name"] == "parents"
            return [], None

    checker = ApplicationReadinessChecker(
        db=_Database(),
        retrieval_settings=RetrievalSettings(backend="qdrant"),
        qdrant_client_factory=lambda: _QdrantClient(),
    )

    with pytest.raises(SetupRequiredError) as exc_info:
        checker.ensure_ready()

    assert exc_info.value.reason == "knowledge_base_empty"

