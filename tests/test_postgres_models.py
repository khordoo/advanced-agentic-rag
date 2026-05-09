from langchain_core.documents import Document
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import configure_mappers

from rag_core.indexing.pgvector_indexer import PgVectorIndexer
from rag_core.schemas import ParentChunk, ParentMetadata
from rag_core.store.postgres.models.documents import DocumentChunk, DocumentSection
from rag_core.store.postgres.models.ingestion import (
    AssetState,
    IngestionRun,
    IngestionRunState,
    IngestionRunStep,
    SourceAsset,
)


def test_postgres_domain_models_configure() -> None:
    configure_mappers()

    assert IngestionRun.__tablename__ == "ingestion_runs"
    assert SourceAsset.__tablename__ == "source_assets"
    assert DocumentSection.__tablename__ == "document_sections"
    assert DocumentChunk.__tablename__ == "document_chunks"


def test_postgres_domain_enums_match_contract() -> None:
    assert {item.name for item in IngestionRunState} == {
        "PENDING",
        "RUNNING",
        "COMPLETED",
        "FAILED",
    }
    assert {item.name for item in IngestionRunStep} == {
        "SCANNING",
        "RECONCILING",
        "INGESTING",
        "FINALIZING",
    }
    assert {item.name for item in AssetState} == {
        "PENDING",
        "PROCESSING",
        "COMPLETED",
        "FAILED",
        "DELETED",
        "EXCLUDED",
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
    assert "asset_key" in DocumentSection.__table__.c
    assert "file_id" not in DocumentSection.__table__.c
    assert "asset_key" in DocumentChunk.__table__.c
    assert "file_id" not in DocumentChunk.__table__.c

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


def test_pgvector_row_builders_use_asset_key() -> None:
    block = ParentChunk(
        id="parent-1",
        enriched_text="parent text",
        metadata=ParentMetadata(
            asset_key="annual_report_a1b2c3d4.pdf",
            file_name="Annual Report.pdf",
            source="/docs/Annual Report.pdf",
            pages=[1],
        ),
        children=[
            Document(
                page_content="child text",
                metadata={
                    "asset_key": "annual_report_a1b2c3d4.pdf",
                    "file_name": "Annual Report.pdf",
                    "pages": [1],
                    "scopes": ["public"],
                    "tags": ["finance"],
                },
            )
        ],
        asset_id=42,
    )

    section = PgVectorIndexer._build_section(block)
    section.id = 123
    chunk = PgVectorIndexer._build_chunk(
        section=section,
        child=block.children[0],
        block=block,
        vector=[0.1, 0.2],
    )

    assert section.asset_key == "annual_report_a1b2c3d4.pdf"
    assert chunk.asset_key == "annual_report_a1b2c3d4.pdf"
    assert "asset_key" not in chunk.chunk_metadata
    assert "file_name" not in chunk.chunk_metadata


def test_source_asset_fencing_and_telemetry_columns() -> None:
    worker_id = SourceAsset.__table__.c.worker_id
    lease_version = SourceAsset.__table__.c.lease_version

    # worker_id is telemetry-only (nullable string)
    assert worker_id.type.length == 255
    assert worker_id.nullable is True

    # lease_version is the fencing token (non-nullable integer, default 0)
    assert lease_version.nullable is False
    assert lease_version.server_default is not None
    assert lease_version.server_default.arg == "0"
