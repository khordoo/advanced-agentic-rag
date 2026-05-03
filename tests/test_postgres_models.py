from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import configure_mappers

from rag_core.store.postgres.models.documents import DocumentChunk, DocumentSection
from rag_core.store.postgres.models.ingestion import (
    AssetStatus,
    IngestionRun,
    IngestionRunPhase,
    IngestionRunStatus,
    SourceAsset,
)


def test_postgres_domain_models_configure() -> None:
    configure_mappers()

    assert IngestionRun.__tablename__ == "ingestion_runs"
    assert SourceAsset.__tablename__ == "source_assets"
    assert DocumentSection.__tablename__ == "document_sections"
    assert DocumentChunk.__tablename__ == "document_chunks"


def test_postgres_domain_enums_match_contract() -> None:
    assert {item.name for item in IngestionRunStatus} == {
        "PENDING",
        "RUNNING",
        "COMPLETED",
        "FAILED",
    }
    assert {item.name for item in IngestionRunPhase} == {
        "SCANNING",
        "PARSING",
        "CHUNKING",
        "EMBEDDING",
        "INGESTING",
        "RECONCILING",
    }
    assert {item.name for item in AssetStatus} == {
        "PENDING",
        "PROCESSING",
        "COMPLETED",
        "FAILED",
        "DELETED",
        "IGNORED",
    }


def test_postgres_domain_foreign_keys_match_contract() -> None:
    source_asset_fk = next(iter(SourceAsset.__table__.c.last_seen_run_id.foreign_keys))
    section_fk = next(iter(DocumentSection.__table__.c.asset_id.foreign_keys))
    chunk_fk = next(iter(DocumentChunk.__table__.c.section_id.foreign_keys))

    assert source_asset_fk.target_fullname == "ingestion_runs.id"
    assert source_asset_fk.ondelete == "SET NULL"
    assert section_fk.target_fullname == "source_assets.id"
    assert section_fk.ondelete == "CASCADE"
    assert chunk_fk.target_fullname == "document_sections.id"
    assert chunk_fk.ondelete == "CASCADE"


def test_postgres_domain_relationships_are_bottom_up_only() -> None:
    assert set(IngestionRun.__mapper__.relationships.keys()) == set()
    assert set(SourceAsset.__mapper__.relationships.keys()) == {"last_seen_run"}
    assert set(DocumentSection.__mapper__.relationships.keys()) == {"asset"}
    assert set(DocumentChunk.__mapper__.relationships.keys()) == {"section"}

    for mapper in (
        SourceAsset.__mapper__,
        DocumentSection.__mapper__,
        DocumentChunk.__mapper__,
    ):
        for relationship in mapper.relationships:
            assert relationship.back_populates is None


def test_postgres_domain_postgres_column_optimizations() -> None:
    array_columns = [
        DocumentSection.__table__.c.page_numbers,
        DocumentChunk.__table__.c.page_numbers,
        DocumentChunk.__table__.c.scopes,
        DocumentChunk.__table__.c.tags,
    ]
    jsonb_columns = [
        SourceAsset.__table__.c.asset_metadata,
        DocumentChunk.__table__.c.chunk_metadata,
    ]

    for column in array_columns:
        assert isinstance(column.type, ARRAY)
        assert column.server_default is not None
        assert column.server_default.arg == "{}"

    for column in jsonb_columns:
        assert isinstance(column.type, JSONB)
        assert column.server_default is not None
        assert column.server_default.arg == "{}"

    assert isinstance(DocumentChunk.__table__.c.vector.type, Vector)
    assert DocumentChunk.__table__.c.vector.type.dim is None


def test_source_asset_worker_telemetry() -> None:
    worker_id = SourceAsset.__table__.c.worker_id

    assert worker_id.type.length == 50
    assert worker_id.nullable is True
