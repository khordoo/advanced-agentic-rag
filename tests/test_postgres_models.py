import pytest
from langchain_core.documents import Document
from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import configure_mappers

from rag_core.indexing.pgvector_indexer import PgVectorIndexer
from rag_core.schemas import ClaimedSourceAsset, ParentChunk, ParentMetadata
from rag_core.store.postgres.models.documents import DocumentChunk, DocumentSection
from rag_core.store.postgres.models.ingestion import (
    AssetState,
    IngestionRun,
    IngestionRunState,
    IngestionRunStep,
    SourceAsset,
)
from rag_core.store.postgres.repositories.documents import DocumentRepository


def _claimed_asset(asset_id: int = 42) -> ClaimedSourceAsset:
    return ClaimedSourceAsset(
        id=asset_id,
        asset_key=f"asset-{asset_id}",
        asset_name=f"asset-{asset_id}.md",
        parent_name="folder",
        source_system="local_fs",
        source_uri=f"file:///asset-{asset_id}.md",
        tags=["test"],
        scopes=["public"],
        asset_metadata={},
        lease_version=1,
    )


def _parent_chunk(
    asset_id: int | None = 42,
    *,
    block_id: str = "parent-1",
) -> ParentChunk:
    return ParentChunk(
        id=block_id,
        enriched_text="parent text",
        metadata=ParentMetadata(
            asset_key=f"asset-{asset_id}",
            file_name=f"asset-{asset_id}.md",
            source=f"file:///asset-{asset_id}.md",
            pages=[1],
        ),
        children=[
            Document(
                page_content="child text",
                metadata={
                    "asset_key": f"asset-{asset_id}",
                    "file_name": f"asset-{asset_id}.md",
                    "pages": [1],
                    "scopes": ["public"],
                    "tags": ["finance"],
                },
            )
        ],
        asset_id=asset_id,
    )


class _FakeDeleteResult:
    def __init__(self, rowcount: int) -> None:
        self.rowcount = rowcount


class _FakeDocumentSession:
    def __init__(self, rowcount: int = 2) -> None:
        self.rowcount = rowcount
        self.executed_stmt = None
        self.events: list[str] = []
        self.add_all_calls: list[list[object]] = []

    def execute(self, stmt):
        self.executed_stmt = stmt
        self.events.append("execute")
        return _FakeDeleteResult(self.rowcount)

    def add_all(self, rows):
        row_list = list(rows)
        self.add_all_calls.append(row_list)
        self.events.append(f"add_all:{len(row_list)}")

    def flush(self):
        self.events.append("flush")
        for index, section in enumerate(self.add_all_calls[0], start=100):
            section.id = index


class _FakeDbSession:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def begin(self):
        return self


class _FakeDb:
    def __init__(self) -> None:
        self.session = _FakeDbSession()

    def get_session(self):
        return self.session


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
    assert "created_at" in DocumentSection.__table__.c
    assert "created_at" in DocumentChunk.__table__.c

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

    timestamp_columns = [
        DocumentSection.__table__.c.created_at,
        DocumentChunk.__table__.c.created_at,
    ]
    for column in timestamp_columns:
        assert isinstance(column.type, DateTime)
        assert column.type.timezone is True
        assert column.nullable is False
        assert column.server_default is not None
        assert column.index is True

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


def test_document_repository_sync_replaces_asset_index() -> None:
    session = _FakeDocumentSession(rowcount=2)
    repo = DocumentRepository(session)
    section = DocumentSection(
        asset_id=77,
        asset_key="asset-77",
        source_uri="file:///asset-77.md",
        doc_type="text",
        page_numbers=[1],
        text="parent text",
    )
    chunk = DocumentChunk(
        section=section,
        asset_key="asset-77",
        page_numbers=[1],
        text="child text",
        vector=[0.1, 0.2],
        scopes=["public"],
        tags=["finance"],
        chunk_metadata={"parent_id": "stale-parent"},
    )

    deleted = repo.sync_document_index(
        asset_id=77,
        sections=[section],
        chunks=[chunk],
    )

    compiled_delete = str(
        session.executed_stmt.compile(compile_kwargs={"literal_binds": True})
    )
    assert deleted == 2
    assert "DELETE FROM document_sections" in compiled_delete
    assert "document_sections.asset_id = 77" in compiled_delete
    assert session.events == ["execute", "add_all:1", "flush", "add_all:1"]
    assert session.add_all_calls == [[section], [chunk]]
    assert chunk.section_id == 100
    assert chunk.chunk_metadata["parent_id"] == 100


def test_document_repository_empty_sync_still_deletes_asset_index() -> None:
    session = _FakeDocumentSession(rowcount=3)
    repo = DocumentRepository(session)

    deleted = repo.sync_document_index(asset_id=91, sections=[], chunks=[])

    compiled_delete = str(
        session.executed_stmt.compile(compile_kwargs={"literal_binds": True})
    )
    assert deleted == 3
    assert "document_sections.asset_id = 91" in compiled_delete
    assert session.events == ["execute", "add_all:0", "flush", "add_all:0"]


def test_pgvector_index_asset_replaces_by_claimed_asset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, list[DocumentSection], list[DocumentChunk]]] = []

    class _FakeRepository:
        def __init__(self, session: _FakeDbSession) -> None:
            self.session = session

        def sync_document_index(
            self,
            asset_id: int,
            sections: list[DocumentSection],
            chunks: list[DocumentChunk],
        ) -> int:
            calls.append((asset_id, sections, chunks))
            return 0

    monkeypatch.setattr(
        "rag_core.indexing.pgvector_indexer.DocumentRepository",
        _FakeRepository,
    )
    indexer = object.__new__(PgVectorIndexer)
    indexer._db = _FakeDb()
    indexer._embed_texts = lambda texts: [[0.1, 0.2] for _ in texts]

    indexer.index_asset(_claimed_asset(42), [_parent_chunk(42)])

    asset_id, sections, chunks = calls[0]
    assert asset_id == 42
    assert len(sections) == 1
    assert len(chunks) == 1
    assert sections[0].asset_id == 42
    assert chunks[0].section is sections[0]


def test_pgvector_index_asset_empty_batch_still_syncs_asset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, list[DocumentSection], list[DocumentChunk]]] = []

    class _FakeRepository:
        def __init__(self, session: _FakeDbSession) -> None:
            self.session = session

        def sync_document_index(
            self,
            asset_id: int,
            sections: list[DocumentSection],
            chunks: list[DocumentChunk],
        ) -> int:
            calls.append((asset_id, sections, chunks))
            return 0

    def fail_embed(texts: list[str]) -> list[list[float]]:
        raise AssertionError("empty asset sync should not embed")

    monkeypatch.setattr(
        "rag_core.indexing.pgvector_indexer.DocumentRepository",
        _FakeRepository,
    )
    indexer = object.__new__(PgVectorIndexer)
    indexer._db = _FakeDb()
    indexer._embed_texts = fail_embed

    indexer.index_asset(_claimed_asset(99), [])

    assert calls == [(99, [], [])]


def test_pgvector_ingest_documents_rejects_missing_asset_id() -> None:
    indexer = object.__new__(PgVectorIndexer)

    with pytest.raises(ValueError, match="missing asset_id"):
        indexer.ingest_documents([_parent_chunk(asset_id=None)])


def test_pgvector_ingest_documents_rejects_mixed_asset_ids() -> None:
    indexer = object.__new__(PgVectorIndexer)

    with pytest.raises(ValueError, match="one asset_id per batch"):
        indexer.ingest_documents(
            [
                _parent_chunk(asset_id=1, block_id="parent-1"),
                _parent_chunk(asset_id=2, block_id="parent-2"),
            ]
        )


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
