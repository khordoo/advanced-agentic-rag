from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError

from rag_core.core.config import RetrievalSettings, get_qdrant_settings, get_retrieval_settings
from rag_core.store.postgres.session import Database, get_database
from rag_core.store.qdrant import QdrantConnectionManager

logger = logging.getLogger(__name__)

SetupReason = Literal["database_unavailable", "schema_not_migrated", "knowledge_base_empty"]

REQUIRED_TABLES: tuple[str, ...] = (
    "ingestion_runs",
    "source_assets",
    "document_sections",
    "document_chunks",
)


@dataclass(frozen=True)
class SetupRequiredError(Exception):
    """Raised when chat cannot run because local setup is incomplete."""

    reason: SetupReason
    message: str

    def __str__(self) -> str:
        return self.message


class ApplicationReadinessChecker:
    """Read-only preflight for chat paths that depend on ingested RAG data."""

    def __init__(
        self,
        *,
        db: Database | None = None,
        retrieval_settings: RetrievalSettings | None = None,
        qdrant_client_factory: Callable | None = None,
    ) -> None:
        self._db = db or get_database()
        self._retrieval_settings = retrieval_settings or get_retrieval_settings()
        self._qdrant_client_factory = qdrant_client_factory or QdrantConnectionManager.get_client

    def ensure_ready(self) -> None:
        """Raise ``SetupRequiredError`` when chat setup is incomplete."""
        self._ensure_schema_ready()
        self._ensure_knowledge_base_ready()

    def _ensure_schema_ready(self) -> None:
        try:
            inspector = inspect(self._db.engine)
            missing = [table for table in REQUIRED_TABLES if not inspector.has_table(table)]
        except SQLAlchemyError as exc:
            logger.warning("readiness_database_unavailable", exc_info=exc)
            raise SetupRequiredError(
                reason="database_unavailable",
                message=(
                    "The database is not reachable. Start the stack with `make up` "
                    "and retry."
                ),
            ) from exc

        if missing:
            raise SetupRequiredError(
                reason="schema_not_migrated",
                message=(
                    "The database schema is not ready. Run `make ingest` to apply "
                    "migrations and ingest documents, or run `make migrate-up` if "
                    "you only want to apply migrations."
                ),
            )

    def _ensure_knowledge_base_ready(self) -> None:
        if self._retrieval_settings.backend == "postgres":
            self._ensure_postgres_documents_exist()
        else:
            self._ensure_qdrant_documents_exist()

    def _ensure_postgres_documents_exist(self) -> None:
        try:
            with self._db.get_session() as session:
                has_chunks = bool(
                    session.scalar(text("SELECT EXISTS (SELECT 1 FROM document_chunks LIMIT 1)"))
                )
        except SQLAlchemyError as exc:
            logger.warning("readiness_postgres_check_failed", exc_info=exc)
            lowered = str(exc).lower()
            if any(signal in lowered for signal in ("does not exist", "undefinedtable")):
                raise SetupRequiredError(
                    reason="schema_not_migrated",
                    message=(
                        "The database schema is not ready. Run `make ingest` to apply "
                        "migrations and ingest documents, or run `make migrate-up` if "
                        "you only want to apply migrations."
                    ),
                ) from exc
            raise SetupRequiredError(
                reason="database_unavailable",
                message=(
                    "The database is not reachable. Start the stack with `make up` "
                    "and retry."
                ),
            ) from exc

        if not has_chunks:
            raise SetupRequiredError(
                reason="knowledge_base_empty",
                message=(
                    "No documents have been ingested yet. Run `make ingest` or start "
                    "the `local-ingestion-runner` deployment from Prefect, then try again."
                ),
            )

    def _ensure_qdrant_documents_exist(self) -> None:
        settings = get_qdrant_settings()
        try:
            client = self._qdrant_client_factory()
            points, _ = client.scroll(
                collection_name=settings.parent_collection_name,
                limit=1,
                with_payload=False,
            )
        except Exception as exc:
            logger.warning("readiness_qdrant_check_failed", exc_info=exc)
            lowered = str(exc).lower()
            if any(signal in lowered for signal in ("not found", "does not exist", "doesn't exist")):
                raise SetupRequiredError(
                    reason="knowledge_base_empty",
                    message=(
                        "No documents have been ingested yet. Run `make ingest` or start "
                        "the `local-ingestion-runner` deployment from Prefect, then try again."
                    ),
                ) from exc
            raise SetupRequiredError(
                reason="database_unavailable",
                message=(
                    "The database or vector store is not reachable. Start the stack with "
                    "`make up` and retry."
                ),
            ) from exc

        if not points:
            raise SetupRequiredError(
                reason="knowledge_base_empty",
                message=(
                    "No documents have been ingested yet. Run `make ingest` or start "
                    "the `local-ingestion-runner` deployment from Prefect, then try again."
                ),
            )

