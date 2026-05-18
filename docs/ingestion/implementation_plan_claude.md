# Domain-Driven Ingestion Refactor — Detailed Implementation Plan

## Assessment of the Original Plan

I reviewed `docs/ingestion/domain-driven-ingestion-refactor-plan.md` against the current codebase. The plan is solid architecturally. Below are the **gaps and improvements** I identified before building the detailed spec.

### Gaps Found & Fixes Applied

| # | Gap | Fix |
|---|-----|-----|
| 1 | `file_id` format mismatch: plan says `8-char-hex` but existing `generate_file_id()` uses `3-digit decimal` salt from content hash. Changing this breaks determinism for already-indexed data. | **Keep the existing `{folder}_{stem}_{NNN}.{ext}` pattern.** Move the function to `fingerprinting.py` but preserve the algorithm. The plan's `8-char-hex` is superseded by what's already shipped. |
| 2 | Plan references `TableMetadata` in `schemas.py:136` but the actual class is named `TableSummary` — `ExtractedTable.seo_metadata` has type `TableMetadata | None` which doesn't exist (dead code). | Fix the type annotation to `TableSummary | None` during Phase 2 schema migration. |
| 3 | Plan doesn't specify the Postgres connection string construction. Alembic's `env.py` already builds it from env vars. | Reuse the same pattern in `connection.py` via `PostgresSettings`. |
| 4 | Plan doesn't mention `rag_core/retriever.py` (legacy Chroma-based retriever still imported in `__init__.py` and tested in `test_retriever.py`). | Leave untouched in Phase 1-2. Phase 3 can deprecate it. |
| 5 | No Alembic migration strategy specified. Tables exist in models but migrations may not be current. | Add explicit step: run `alembic revision --autogenerate` after Phase 1 models are stable. |
| 6 | Plan says `IngestionSettings.ALLOWED_EXTENSIONS` but doesn't add `INGESTION_ROOT` setting for the scanner. | Add `ingestion_root: str = "data/source_docs"` to `IngestionSettings`. |
| 7 | No `conftest.py` fixtures for Postgres integration tests (in-memory SQLite or test containers). | Use SQLite in-memory for repository unit tests; document Postgres-required tests separately. |
| 8 | `conversion.py` still uses `file_name` key internally in `_enrich_tables_with_surrounding_text`. Plan's Phase 2 text enrichment amendment must update this. | Addressed in Phase 2 conversion.py spec. |

---

## Phase 1: File Lifecycle & State Machine

### Goal

Postgres-only file lifecycle tracking: scanner, advisory locking, reconciliation, canonical `file_id`. **No content parsing, no Qdrant writes, no generation changes.**

---

### 1.1 Config Changes

#### [MODIFY] [config.py](file:///Users/mahmoodkhordoo/Developer/playgorund/rag-chunking/rag_core/core/config.py)

Add to `IngestionSettings`:

```python
ingestion_root: str = Field(
    default="data/source_docs",
    description="Root directory the scanner walks for source documents.",
)
```

Add cached accessor:

```python
@lru_cache()
def get_security_settings() -> SecuritySettings:
    return SecuritySettings()
```

---

### 1.2 Postgres Connection & Unit of Work

#### [NEW] [connection.py](file:///Users/mahmoodkhordoo/Developer/playgorund/rag-chunking/rag_core/store/postgres/connection.py)

```python
"""SQLAlchemy engine/session factory using PostgresSettings."""

def get_engine(settings: PostgresSettings | None = None) -> Engine:
    """Create or return cached SQLAlchemy engine."""

def get_session_factory(engine: Engine | None = None) -> sessionmaker[Session]:
    """Return a session factory bound to the engine."""
```

- Build URL as `postgresql+psycopg://{user}:{password}@{host}:{port}/{rag_db}`
- Pool: `pool_size=5`, `max_overflow=10`, `pool_pre_ping=True`
- Engine cached via module-level singleton (not `lru_cache` — needs to work with tests)

#### [NEW] [unit_of_work.py](file:///Users/mahmoodkhordoo/Developer/playgorund/rag-chunking/rag_core/store/postgres/unit_of_work.py)

```python
class PostgresUnitOfWork:
    """Transaction-scoped session lifecycle with repository accessors."""

    def __init__(self, session_factory=None): ...

    def __enter__(self) -> "PostgresUnitOfWork": ...
    def __exit__(self, exc_type, exc_val, exc_tb): ...
        # Rollback on exception, commit otherwise

    @property
    def pipeline_runs(self) -> PipelineRunRepository: ...

    @property
    def source_assets(self) -> SourceAssetRepository: ...

    # Phase 2 additions:
    # @property
    # def document_sections(self) -> DocumentSectionRepository: ...
    # @property
    # def document_chunks(self) -> DocumentChunkRepository: ...
```

- Lazy-initializes repositories on first access
- `__exit__` does `session.rollback()` on exception, `session.commit()` otherwise
- Accepts optional `session_factory` for test injection

---

### 1.3 Repositories

#### [NEW] [pipeline_runs.py](file:///Users/mahmoodkhordoo/Developer/playgorund/rag-chunking/rag_core/store/postgres/repositories/pipeline_runs.py)

```python
class PipelineRunRepository:
    def __init__(self, session: Session): ...

    def create_ingestion_run(self) -> PipelineRun:
        """Insert a new run with phase=SCANNING, pipeline_type=INGESTION."""

    def set_phase(self, run_id: int, phase: RunPhase) -> None:
        """Update the phase column."""

    def complete(self, run_id: int) -> None:
        """Set phase=COMPLETED, completed_at=now()."""

    def fail(self, run_id: int, error_message: str) -> None:
        """Set phase=FAILED, completed_at=now(), error_message."""

    def try_acquire_scanner_lock(self) -> bool:
        """SELECT pg_try_advisory_lock(hashtext('ingestion_scanner'))"""

    def release_scanner_lock(self) -> None:
        """SELECT pg_advisory_unlock(hashtext('ingestion_scanner'))"""
```

#### [NEW] [source_assets.py](file:///Users/mahmoodkhordoo/Developer/playgorund/rag-chunking/rag_core/store/postgres/repositories/source_assets.py)

```python
class SourceAssetRepository:
    def __init__(self, session: Session): ...

    def get_by_file_id(self, file_id: str) -> SourceAsset | None: ...

    def upsert_seen_asset(
        self,
        file_id: str,
        folder_name: str,
        file_name: str,
        source_uri: str,
        content_hash: str,
        run_id: int,
        worker_id: str | None = None,
    ) -> SourceAsset:
        """
        - New file → INSERT as PENDING
        - Existing unchanged → update last_seen_run_id, updated_at
        - Existing changed (content_hash differs) → update hash, set PENDING
        - Existing DELETED → set PENDING
        """

    def mark_ignored(
        self, file_id: str, run_id: int, reason: str,
        folder_name: str, file_name: str, source_uri: str,
    ) -> None:
        """Upsert as IGNORED with reason in asset_metadata."""

    def mark_missing_assets_deleted(
        self, current_run_id: int, batch_size: int = 1000
    ) -> int:
        """
        UPDATE source_assets SET status='deleted'
        WHERE last_seen_run_id != current_run_id AND status NOT IN ('deleted','ignored')
        Process in batches. Return total marked.
        """
```

#### [NEW] [repositories/__init__.py](file:///Users/mahmoodkhordoo/Developer/playgorund/rag-chunking/rag_core/store/postgres/repositories/__init__.py)

```python
from .pipeline_runs import PipelineRunRepository
from .source_assets import SourceAssetRepository

__all__ = ["PipelineRunRepository", "SourceAssetRepository"]
```

---

### 1.4 Scanner

#### [NEW] [scanner.py](file:///Users/mahmoodkhordoo/Developer/playgorund/rag-chunking/rag_core/ingestion/scanner.py)

```python
@dataclass
class DiscoveredFile:
    absolute_path: Path
    relative_path: Path
    parent_folder: str
    file_stem: str
    extension: str
    original_file_name: str
    source_uri: str          # str(absolute_path)

@dataclass
class IgnoredFile:
    absolute_path: Path
    relative_path: Path
    original_file_name: str
    reason: str              # e.g. "unsupported_extension", "ignored_filename", "hidden_file"

@dataclass
class ScanResult:
    supported: list[DiscoveredFile]
    ignored: list[IgnoredFile]

def scan_directory(
    ingestion_root: Path,
    allowed_extensions: set[str],
    ignore_dirs: set[str] | None = None,
    ignore_files: set[str] | None = None,
) -> ScanResult:
    """Walk ingestion_root, classify files into supported/ignored."""
```

- Defaults for `ignore_dirs`: `{".DS_Store", "__pycache__", ".git"}`
- Defaults for `ignore_files`: `{"README.md", "LICENSE", ".gitignore"}`
- Hidden files (starting with `.`) → `IgnoredFile` with reason `"hidden_file"`
- Unsupported extensions → `IgnoredFile` with reason `"unsupported_extension"`

---

### 1.5 Fingerprinting

#### [NEW] [fingerprinting.py](file:///Users/mahmoodkhordoo/Developer/playgorund/rag-chunking/rag_core/ingestion/fingerprinting.py)

Move `generate_file_id` and `compute_content_hash` from `rag_core/utils.py` to here. Keep the **existing algorithm** (`{folder}_{stem}_{NNN}.{ext}` with 3-digit decimal salt).

```python
def build_file_id(file_path: Path, ingestion_root: Path) -> str:
    """Canonical file_id. Delegates to existing generate_file_id logic."""

def compute_content_hash(file_path: Path) -> str:
    """SHA-256 of file bytes."""
```

- `rag_core/utils.py` retains thin re-exports for backward compat:
  ```python
  from rag_core.ingestion.fingerprinting import build_file_id as generate_file_id
  from rag_core.ingestion.fingerprinting import compute_content_hash
  ```

---

### 1.6 Ingestion State

#### [NEW] [state.py](file:///Users/mahmoodkhordoo/Developer/playgorund/rag-chunking/rag_core/ingestion/state.py)

```python
"""Thin re-exports of state enums for ingestion domain code."""
from rag_core.store.postgres.models.documents import AssetStatus
from rag_core.store.postgres.models.pipelines import RunPhase, PipelineType
```

---

### 1.7 Lifecycle Pipeline

#### [NEW] [pipeline.py](file:///Users/mahmoodkhordoo/Developer/playgorund/rag-chunking/rag_core/ingestion/pipeline.py)

```python
class IngestionLifecyclePipeline:
    """Phase 1: File discovery, fingerprinting, and Postgres lifecycle tracking."""

    def __init__(self, settings: IngestionSettings | None = None,
                 uow_factory=None): ...

    def run(self) -> None:
        """
        1. Create PipelineRun
        2. Acquire advisory lock (abort if already held)
        3. Set phase → SCANNING
        4. Scan files via scanner.scan_directory()
        5. Set phase → RECONCILING
        6. For each supported file:
           - build_file_id + compute_content_hash
           - uow.source_assets.upsert_seen_asset(...)
        7. For each ignored file:
           - uow.source_assets.mark_ignored(...)
        8. Mark missing assets as DELETED in 1000-row batches
        9. Set phase → COMPLETED
        On exception: mark run FAILED
        Finally: release advisory lock
        """
```

#### [NEW] [__init__.py](file:///Users/mahmoodkhordoo/Developer/playgorund/rag-chunking/rag_core/ingestion/__init__.py)

```python
from rag_core.ingestion.pipeline import IngestionLifecyclePipeline
```

---

### 1.8 Phase 1 Tests

#### [NEW] [test_ingestion_lifecycle.py](file:///Users/mahmoodkhordoo/Developer/playgorund/rag-chunking/tests/test_ingestion_lifecycle.py)

Uses SQLite in-memory via `conftest.py` fixture (advisory lock tests skipped on SQLite, marked `@pytest.mark.postgres`).

| Test | What it verifies |
|------|-----------------|
| `test_file_id_uses_parent_folder_stem_ext_and_salt` | Format matches `{folder}_{stem}_{NNN}.{ext}` |
| `test_identical_names_different_folders_produce_different_ids` | Uniqueness across folders |
| `test_scanner_emits_supported_and_ignored` | Scanner classifies correctly |
| `test_scanner_reports_ignore_reasons` | Each ignored file has a reason string |
| `test_lifecycle_creates_pipeline_run` | Run row inserted with SCANNING phase |
| `test_lifecycle_upserts_new_files_as_pending` | New assets are PENDING |
| `test_lifecycle_updates_last_seen_run_id` | Unchanged files get updated run reference |
| `test_lifecycle_marks_changed_content_pending` | Different hash → PENDING |
| `test_lifecycle_marks_missing_as_deleted` | Assets not in scan → DELETED |
| `test_lifecycle_marks_ignored_files` | Ignored assets get IGNORED status + reason |
| `test_reconciliation_batches_deletes` | Batched at 1000 rows |
| `test_scanner_aborts_on_lock_failure` | `@pytest.mark.postgres` — controlled error |
| `test_failed_run_marks_phase_failed` | Exception → FAILED + error_message |

#### [NEW] [test_postgres_repositories.py](file:///Users/mahmoodkhordoo/Developer/playgorund/rag-chunking/tests/test_postgres_repositories.py)

| Test | What it verifies |
|------|-----------------|
| `test_create_ingestion_run` | Inserts run, returns model with id |
| `test_set_phase` | Phase column updated |
| `test_complete_run` | Phase=COMPLETED, completed_at set |
| `test_fail_run` | Phase=FAILED, error_message stored |
| `test_upsert_new_asset` | Creates PENDING asset |
| `test_upsert_existing_unchanged` | Only updates last_seen_run_id |
| `test_upsert_existing_changed_hash` | Resets to PENDING |
| `test_upsert_deleted_asset_reactivates` | DELETED → PENDING |
| `test_mark_ignored` | Status=IGNORED, reason in metadata |
| `test_mark_missing_deleted_batches` | Processes in batches, returns count |

#### [NEW] [conftest.py](file:///Users/mahmoodkhordoo/Developer/playgorund/rag-chunking/tests/conftest.py)

```python
@pytest.fixture
def sqlite_session():
    """In-memory SQLite session for unit-testing repositories."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()

@pytest.fixture
def uow(sqlite_session):
    """UoW wired to SQLite for pure unit tests."""
    ...
```

---

## Phase 2: Schema Package Refactor & Content Ingestion

### Goal

Replace monolithic `schemas.py` with modular package. Parse documents, write `DocumentSection` and `DocumentChunk` to Postgres. **Qdrant untouched.**

---

### 2.1 Schema Package

#### [RENAME] `rag_core/schemas.py` → [legacy_schema.py](file:///Users/mahmoodkhordoo/Developer/playgorund/rag-chunking/rag_core/legacy_schema.py)

Copy contents verbatim. Fix `ExtractedTable.seo_metadata` type from `TableMetadata` → `TableSummary`.

#### [NEW] `rag_core/schemas/` — Package with these modules:

| Module | Contents |
|--------|----------|
| `documents.py` | `SourceAssetSchema`, `DocumentSectionSchema`, `DocumentChunkSchema`, `DocumentChunkMetadata`, `DocumentSectionMetadata` |
| `pipelines.py` | `PipelineRunSchema`, `PipelineRunCreate`, `PipelineRunUpdate` |
| `ingestion.py` | `DiscoveredFile`, `IgnoredFile`, `FileFingerprint`, `IngestionResult`, `AssetLifecycleDecision` |
| `tables.py` | `TableSummary`, `TableChunk`, `ExtractedTable` (moved from legacy) |
| `prompts.py` | `PromptConfig` (moved from legacy) |
| `evaluation.py` | `QAPair`, `SyntheticsTestSet`, `QARecord` (moved from legacy) |
| `qdrant.py` | `VectorDatabasePayloadKeys`, `PARENT_INDEX_SCHEMA`, `CHILD_INDEX_SCHEMA`, `ParentChunk`, `BaseChunkMetadata`, `ParentMetadata`, `ChildMetadata` (legacy Qdrant types until Phase 3) |

#### [NEW] `rag_core/schemas/__init__.py` — Compatibility Exports

```python
# Legacy re-exports — all existing `from rag_core.schemas import X` continue to work
from rag_core.schemas.evaluation import QAPair, SyntheticsTestSet, QARecord
from rag_core.schemas.prompts import PromptConfig
from rag_core.schemas.tables import TableSummary, TableChunk, ExtractedTable
from rag_core.schemas.qdrant import (
    ParentChunk, BaseChunkMetadata, ParentMetadata, ChildMetadata,
    VectorDatabasePayloadKeys, PARENT_INDEX_SCHEMA, CHILD_INDEX_SCHEMA,
)
```

### 2.2 Files to Update for Schema Imports

These files import from `rag_core.schemas` — they will continue to work via `__init__.py` re-exports. **No import changes required in Phase 2** for:
- `rag_core/chunker.py`
- `rag_core/agents.py`
- `rag_core/indexing/base.py`, `qdrant_indexer.py`
- `rag_core/retrival/qdrant_retriever.py`
- `rag_core/genetation/tools.py`
- `rag_core/__init__.py`

Update `rag_core/__init__.py` to import from `rag_core.schemas` (still works via re-exports).

---

### 2.3 Content Ingestion Modules

#### [NEW] [conversion.py](file:///Users/mahmoodkhordoo/Developer/playgorund/rag-chunking/rag_core/ingestion/conversion.py)

```python
def convert_to_markdown(
    file_path: Path, file_id: str, table_context_window: int = 3
) -> tuple[str, list[dict]]:
    """Wraps existing converter for PDF/MD, returns (markdown, enriched_tables).
    Delegates to rag_core.converter internally.
    """
```

#### [NEW] [sectioning.py](file:///Users/mahmoodkhordoo/Developer/playgorund/rag-chunking/rag_core/ingestion/sectioning.py)

```python
def create_sections_and_chunks(
    markdown: str,
    file_id: str,
    source_uri: str,
    asset_id: int,
    tags: list[str],
    scopes: list[str],
    config: dict | None = None,
) -> tuple[list[DocumentSectionSchema], list[DocumentChunkSchema]]:
    """
    Maps markdown → validated DocumentSectionSchema + DocumentChunkSchema.
    Uses existing chunker logic internally but outputs domain schemas.
    Injects file_id into all text headers (not file_name).
    """
```

#### [NEW] [writer.py](file:///Users/mahmoodkhordoo/Developer/playgorund/rag-chunking/rag_core/ingestion/writer.py)

```python
class ContentWriter:
    """Writes sections/chunks to Postgres, replacing stale data in one transaction."""

    def __init__(self, uow: PostgresUnitOfWork): ...

    def write_for_asset(
        self, asset_id: int, file_id: str,
        sections: list[DocumentSectionSchema],
        chunks: list[DocumentChunkSchema],
    ) -> None:
        """
        1. DELETE existing sections/chunks for this asset_id (CASCADE handles chunks)
        2. INSERT new sections
        3. INSERT new chunks with section_id FK
        All within the UoW's transaction.
        """
```

### 2.4 New Repositories (Phase 2)

#### [NEW] [document_sections.py](file:///Users/mahmoodkhordoo/Developer/playgorund/rag-chunking/rag_core/store/postgres/repositories/document_sections.py)

```python
class DocumentSectionRepository:
    def bulk_insert(self, sections: list[DocumentSection]) -> list[int]: ...
    def delete_by_asset_id(self, asset_id: int) -> int: ...
```

#### [NEW] [document_chunks.py](file:///Users/mahmoodkhordoo/Developer/playgorund/rag-chunking/rag_core/store/postgres/repositories/document_chunks.py)

```python
class DocumentChunkRepository:
    def bulk_insert(self, chunks: list[DocumentChunk]) -> list[int]: ...
    def delete_by_section_ids(self, section_ids: list[int]) -> int: ...
```

### 2.5 Phase 2 Tests

| Test file | Key tests |
|-----------|-----------|
| `test_schema_package_imports.py` | `from rag_core.schemas import ParentChunk` still works; `from rag_core.legacy_schema import ...` works; new modules validate fields |
| `test_ingestion_content_write.py` | Content write creates sections + chunks; re-ingesting replaces old rows; text headers contain `file_id`; `chunk_metadata` stores `file_id` |

---

## Phase 3: Vector Store & Retrieval Migration

### Goal

Migrate Qdrant indexing and retrieval from legacy `ParentChunk`/`file_name` to `DocumentChunkSchema`/`file_id`.

---

### 3.1 Indexing Changes

#### [MODIFY] [base.py](file:///Users/mahmoodkhordoo/Developer/playgorund/rag-chunking/rag_core/indexing/base.py)

- `ingest_documents()` accepts `list[DocumentChunkSchema]` (with legacy overload or union type)

#### [MODIFY] [qdrant_indexer.py](file:///Users/mahmoodkhordoo/Developer/playgorund/rag-chunking/rag_core/indexing/qdrant_indexer.py)

- Payload writes `file_id` instead of `file_name`
- New keys: `section_id`, `asset_id`
- `VectorDatabasePayloadKeys.FILE_NAME` → `FILE_ID`

### 3.2 Retrieval & Generation Changes

#### [MODIFY] [qdrant_retriever.py](file:///Users/mahmoodkhordoo/Developer/playgorund/rag-chunking/rag_core/retrival/qdrant_retriever.py)

- Filters accept `file_id` instead of `file_name`
- `_build_filters` uses `FILE_ID` key

#### [MODIFY] [tools.py](file:///Users/mahmoodkhordoo/Developer/playgorund/rag-chunking/rag_core/genetation/tools.py)

- `TableLookupToolSchema.file_name` → `file_id`
- `FetchPageToolSchema.file_name` → `file_id`
- Internal filter calls updated

### 3.3 Cleanup

- `rag_core/schemas/qdrant.py` removes `FILE_NAME` from payload enum
- `rag_core/legacy_schema.py` kept but no production imports remain
- `rag_core/schemas/__init__.py` compatibility exports reviewed/reduced

### 3.4 Phase 3 Tests

| Test | What it verifies |
|------|-----------------|
| Indexer accepts new chunk schemas, writes `file_id` payloads |
| Retrieval filters reject legacy `file_name` |
| Page lookup uses `file_id` |
| Generation tools expose `file_id` in arg schemas |
| No production code imports from `legacy_schema` |

---

## Execution Order

```
Phase 1 → commit → pytest + ruff → review
Phase 2 → commit → full test suite (verify Qdrant tests still pass) → review
Phase 3 → commit → full integration test → review
```

## Verification Plan

### Automated Tests
- `uv run pytest tests/` after each phase
- `uv run ruff check rag_core/` after each phase
- Phase-specific test files validate new behavior

### Manual Verification
- Phase 1: Run lifecycle pipeline against local Postgres, inspect `pipeline_runs` and `source_assets` tables
- Phase 2: Verify `document_sections` and `document_chunks` populated correctly
- Phase 3: Full end-to-end ingestion + retrieval + generation test

## Open Questions

> [!IMPORTANT]
> **Q1: `file_id` salt format** — The original plan specifies an `8-char-hex` salt but the existing codebase uses a `3-digit decimal` salt. I recommend keeping the existing format to avoid breaking already-indexed data. Do you agree?

> [!IMPORTANT]
> **Q2: Postgres test strategy** — For unit tests, I'll use SQLite in-memory (advisory lock tests marked `@pytest.mark.postgres`). Do you want a Docker-based integration test suite as well, or is SQLite sufficient for now?

> [!IMPORTANT]
> **Q3: Phase execution** — Do you want me to implement all three phases, or start with Phase 1 only and get your review before proceeding?
