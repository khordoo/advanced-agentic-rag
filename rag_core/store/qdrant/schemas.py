"""Qdrant-specific payload index schema definitions.

These constants depend on ``qdrant_client`` and are isolated here so that
the core ``rag_core.schemas`` module remains importable without the
optional Qdrant dependency (e.g. during Alembic migrations).

The ``VectorDatabasePayloadKeys`` enum lives in ``rag_core.schemas``
because it is shared across both Qdrant and Postgres stores.
"""

from __future__ import annotations

from qdrant_client import models

from rag_core.schemas import VectorDatabasePayloadKeys

_BASE_INDEX_SCHEMA = {
    VectorDatabasePayloadKeys.ASSET_KEY.value: models.PayloadSchemaType.KEYWORD,
    VectorDatabasePayloadKeys.ASSET_NAME.value: models.PayloadSchemaType.KEYWORD,
    VectorDatabasePayloadKeys.CREATED_AT.value: models.PayloadSchemaType.DATETIME,
    VectorDatabasePayloadKeys.SOURCE.value: models.PayloadSchemaType.KEYWORD,
    VectorDatabasePayloadKeys.PAGE.value: models.PayloadSchemaType.INTEGER,
    VectorDatabasePayloadKeys.SCOPES.value: models.PayloadSchemaType.KEYWORD,
    VectorDatabasePayloadKeys.TAGS.value: models.PayloadSchemaType.KEYWORD,
}

PARENT_INDEX_SCHEMA = {
    **_BASE_INDEX_SCHEMA,
    VectorDatabasePayloadKeys.TABLE_ID.value: models.PayloadSchemaType.KEYWORD,
}

CHILD_INDEX_SCHEMA = {
    **_BASE_INDEX_SCHEMA,
    VectorDatabasePayloadKeys.PARENT_ID.value: models.PayloadSchemaType.KEYWORD,
}

__all__ = [
    "CHILD_INDEX_SCHEMA",
    "PARENT_INDEX_SCHEMA",
    "_BASE_INDEX_SCHEMA",
]
