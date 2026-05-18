# Domain-Driven Ingestion Refactor Plan

## Summary

This document captures the full three-phase architectural refactor for the RAG ingestion pipeline.

The target architecture moves ingestion toward Domain-Driven Design while protecting the existing Qdrant vector store and generation code until the final migration phase.

Core principles:

- Postgres is the source of truth for pipeline runs, source assets, document sections, and document chunks.
- `file_id` is the canonical durable identifier for all new code.
- `file_name` remains human-readable provenance only.
- Existing Qdrant indexing, retrieval, and generation behavior must remain stable until Phase 3.
- Refactor in strict sequential phases to avoid breaking downstream systems prematurely.

## Target Directory Structure

```text
rag_core/
  ingestion/
    __init__.py
    pipeline.py
    scanner.py
    fingerprinting.py
    metadata.py
    state.py
    conversion.py
    sectioning.py
    writer.py

  schemas/
    __init__.py
    documents.py
    pipelines.py
    ingestion.py
    evaluation.py
    prompts.py
    tables.py
    qdrant.py

  legacy_schema.py

  store/
    postgres/
      connection.py
      unit_of_work.py
      repositories/
        __init__.py
        pipeline_runs.py
        source_assets.py
        document_sections.py
        document_chunks.py
      models/
        base.py
        documents.py
        pipelines.py

  pipelines/
    ingestion.py
    pipeline.py
```

## Phase 1: File Lifecycle & State Machine

### Goal

Implement Postgres-only file lifecycle tracking with a scanner heartbeat, advisory locking, high-water mark reconciliation, and stable canonical `file_id` generation.

Phase 1 does not parse content, generate chunks, write vectors, call Qdrant, or touch generation/retrieval logic.

### Key Decisions

- Scanner lock: use `pg_try_advisory_lock(hashtext('ingestion_scanner'))`.
- If the lock cannot be acquired, abort the run with a controlled "scanner already running" error.
- File identity pattern: `{parent_folder}_{file_stem}_{8-char-hex}.{ext}`.
- The `8-char-hex` fingerprint is derived from the hash of the file's full relative path from the ingestion root.
- `file_name` remains human-readable provenance only.
- `file_id` is the canonical durable identifier.

### Files To Create

- `rag_core/ingestion/__init__.py`
- `rag_core/ingestion/scanner.py`
- `rag_core/ingestion/fingerprinting.py`
- `rag_core/ingestion/metadata.py`
- `rag_core/ingestion/state.py`
- `rag_core/ingestion/pipeline.py`
- `rag_core/store/postgres/connection.py`
- `rag_core/store/postgres/unit_of_work.py`
- `rag_core/store/postgres/repositories/__init__.py`
- `rag_core/store/postgres/repositories/pipeline_runs.py`
- `rag_core/store/postgres/repositories/source_assets.py`
- `tests/test_ingestion_lifecycle.py`
- `tests/test_postgres_repositories.py`

### Files To Modify

- `rag_core/core/config.py`
- `rag_core/pipelines/ingestion.py` only if a compatibility wrapper is needed.
- Do not modify `rag_core/chunker.py`, `rag_core/indexing/*`, `rag_core/retrival/*`, `rag_core/converter.py`, or generation code in Phase 1.

### Postgres Connection & Unit Of Work

Implement:

- SQLAlchemy engine/session factory using `PostgresSettings`.
- `PostgresUnitOfWork` with transaction-scoped session lifecycle.
- Repository accessors:
  - `pipeline_runs`
  - `source_assets`
- Rollback on exceptions.
- Advisory lock cleanup even on run failure.

### PipelineRun Repository

Implement:

- `create_ingestion_run() -> PipelineRun`
- `set_phase(run_id, RunPhase)`
- `complete(run_id)`
- `fail(run_id, error_message)`
- `try_acquire_scanner_lock() -> bool`
- `release_scanner_lock()`

Behavior:

- `try_acquire_scanner_lock()` executes `SELECT pg_try_advisory_lock(hashtext('ingestion_scanner'))`.
- If lock acquisition fails, no scan/reconciliation work runs.
- `complete()` sets `phase=COMPLETED` and `completed_at`.
- `fail()` sets `phase=FAILED`, `completed_at`, and `error_message`.

### SourceAsset Repository

Implement:

- `get_by_file_id(file_id)`
- `upsert_seen_asset(discovered_file, run_id, worker_id=None)`
- `mark_ignored(discovered_file, run_id, reason)`
- `mark_missing_assets_deleted(current_run_id, batch_size=1000)`

Behavior:

- Every supported file seen on disk updates:
  - `last_seen_run_id`
  - `updated_at`
  - `folder_name`
  - `file_name`
  - `source_uri`
  - `content_hash`
  - `worker_id` if provided
- New supported files are inserted as `PENDING`.
- Existing unchanged files preserve current status except `DELETED`, which becomes `PENDING`.
- Existing changed files update `content_hash` and become `PENDING`.
- Ignored files become `IGNORED` and store the reason in `asset_metadata`.
- Missing assets are marked `DELETED` in batches of `1,000`.

### Scanner

Implement in `rag_core/ingestion/scanner.py`:

- Walk the configured ingestion root.
- Skip hidden/system directories and files consistently with current behavior.
- Emit supported files and ignored files separately.
- Supported extensions come from `IngestionSettings.ALLOWED_EXTENSIONS`.
- Ignore rules include current ignored names:
  - directories: `.DS_Store`, `__pycache__`, `.git`
  - files: `README.md`, `LICENSE`, `.gitignore`
  - hidden files

Scanner output should include:

- absolute path
- relative path
- parent folder
- file stem
- extension
- original file name
- source URI
- ignore reason when ignored

### Fingerprinting

Implement in `rag_core/ingestion/fingerprinting.py`:

- `build_file_id(path, ingestion_root) -> str`
- `compute_content_hash(path) -> str`

Rules:

- `file_id` pattern is `{parent_folder}_{file_stem}_{8-char-hex}.{ext}`.
- The `8-char-hex` is derived from the full relative path hash.
- Content hash is derived from file bytes.
- Normalize path separators so IDs are stable across platforms.

### Lifecycle Pipeline

Implement in `rag_core/ingestion/pipeline.py`:

- Create `PipelineRun`.
- Acquire scanner advisory lock using try-lock.
- Set phase to `SCANNING`.
- Scan supported and ignored files.
- Set phase to `RECONCILING`.
- Upsert every supported file with current `last_seen_run_id`.
- Mark ignored files with current `last_seen_run_id`.
- Mark assets not seen in current run as `DELETED` in 1,000-row batches.
- Mark run `COMPLETED`.
- On exception, mark run `FAILED`.
- Always release advisory lock if acquired.

### Phase 1 Tests

Create tests for:

- `file_id` uses parent folder, file stem, extension, and 8-char hex relative-path fingerprint.
- Identical names in different folders produce different `file_id`s.
- Scanner emits supported files and ignored files with reasons.
- Lifecycle creates a `PipelineRun`.
- Lifecycle upserts new files as `PENDING`.
- Lifecycle updates `last_seen_run_id` for unchanged files.
- Lifecycle marks changed content as `PENDING`.
- Lifecycle marks missing files as `DELETED`.
- Lifecycle marks ignored files as `IGNORED`.
- Reconciliation processes deleted assets in 1,000-row batches.
- Scanner aborts when advisory try-lock returns false.
- Failed lifecycle marks `PipelineRun.phase = FAILED`.

## Phase 2: Pydantic Schema Package Refactor & Content Ingestion

### Goal

Replace the monolithic `rag_core/schemas.py` file with a modular schema package, parse source documents, validate the parsed structure, and write `DocumentSection` and `DocumentChunk` rows into Postgres.

Phase 2 keeps Qdrant indexing and generation behavior stable.

### Files To Create

- `rag_core/schemas/__init__.py`
- `rag_core/schemas/documents.py`
- `rag_core/schemas/pipelines.py`
- `rag_core/schemas/ingestion.py`
- `rag_core/schemas/evaluation.py`
- `rag_core/schemas/prompts.py`
- `rag_core/schemas/tables.py`
- `rag_core/schemas/qdrant.py`
- `rag_core/ingestion/conversion.py`
- `rag_core/ingestion/sectioning.py`
- `rag_core/ingestion/writer.py`
- `rag_core/store/postgres/repositories/document_sections.py`
- `rag_core/store/postgres/repositories/document_chunks.py`
- `tests/test_schema_package_imports.py`
- `tests/test_ingestion_content_write.py`

### Files To Rename

- `rag_core/schemas.py` -> `rag_core/legacy_schema.py`

### Files To Modify

- `rag_core/__init__.py`
- `rag_core/agents.py`
- `rag_core/chunker.py`
- `rag_core/indexing/base.py`
- `rag_core/indexing/qdrant_indexer.py`
- `rag_core/retrival/qdrant_retriever.py`
- `rag_core/genetation/tools.py`
- `tests/test_chunker.py`
- `tests/test_ingestion_pipeline.py`

### Compatibility Rule

During Phase 2:

- `rag_core/schemas/__init__.py` must re-export legacy symbols needed by existing Qdrant/generation code.
- Existing imports like `from rag_core.schemas import ParentChunk` must continue to work until Phase 3.
- New domain code should import explicit modules such as `rag_core.schemas.documents`.

### Proposed Schema Package Contents

- `documents.py`
  - `SourceAssetSchema`
  - `DocumentSectionSchema`
  - `DocumentChunkSchema`
  - `DocumentChunkMetadata`
  - `DocumentSectionMetadata`

- `pipelines.py`
  - `PipelineRunSchema`
  - `PipelineRunCreate`
  - `PipelineRunUpdate`

- `ingestion.py`
  - `DiscoveredFile`
  - `IgnoredFile`
  - `FileFingerprint`
  - `IngestionResult`
  - `AssetLifecycleDecision`

- `tables.py`
  - `TableSummary`
  - `TableChunk`
  - `ExtractedTable`

- `prompts.py`
  - `PromptConfig`

- `evaluation.py`
  - `QAPair`
  - `SyntheticsTestSet`
  - `QARecord`

- `qdrant.py`
  - temporary legacy payload keys and Qdrant index schemas until Phase 3 removes or renames them.

### Content Ingestion Behavior

- `conversion.py` wraps PDF and Markdown conversion into a canonical markdown result.
- `sectioning.py` maps markdown output into validated `DocumentSectionSchema` and `DocumentChunkSchema`.
- `writer.py` deletes/replaces sections and chunks for changed assets inside one transaction.
- `DocumentSection.asset_id` links to `SourceAsset.id`.
- `DocumentChunk.section_id` links to `DocumentSection.id`.
- `file_id` is written to every section and chunk.
- Existing Qdrant indexer is not called by the new Postgres writer in Phase 2.

### Text Enrichment Amendment

When updating chunking and sectioning logic:

- Replace all text-level injected identifiers from `file_name` to canonical `file_id`.
- Headers injected into section/chunk text must use `file_id`, for example `Source: {file_id}, Page: {n}`.
- Child chunk text must include `file_id` because the LLM sees retrieved text directly.
- Parent metadata injected into child chunks must carry `file_id`.
- The final `DocumentChunk.chunk_metadata` JSONB payload must carry `file_id`.
- This is required so Phase 3 retrieval can filter by `file_id` and so the LLM agent can call follow-up search/filtering tools with the correct identifier.

### Phase 2 Tests

Create or update tests for:

- `rag_core.legacy_schema` exposes old schema objects.
- `rag_core.schemas` compatibility exports keep current imports working.
- New schema modules validate `file_id`, page numbers, text, scopes, tags, and metadata.
- Content write creates sections and chunks for a changed asset.
- Re-ingesting a changed asset replaces old sections/chunks without duplicating rows.
- Text-level section/chunk headers contain `file_id`, not `file_name`.
- Child chunk metadata enrichment carries `file_id`.
- `DocumentChunk.chunk_metadata` stores `file_id`.
- Existing Qdrant-related tests still pass unchanged.

## Phase 3: Vector Store & Retrieval Migration

### Goal

Move Qdrant indexing and retrieval to the new document-domain schemas and `file_id` metadata.

Remove dependence on legacy `ParentChunk`, `ParentMetadata`, `ChildMetadata`, and `file_name` as the source identifier.

### Files To Modify

- `rag_core/indexing/base.py`
- `rag_core/indexing/qdrant_indexer.py`
- `rag_core/retrival/qdrant_retriever.py`
- `rag_core/genetation/tools.py`
- `rag_core/pipelines/ingestion.py`
- `rag_core/pipelines/pipeline.py`
- `rag_core/chunker.py`
- `rag_core/schemas/qdrant.py`
- `tests/test_retriever.py`
- `tests/test_ingestion_pipeline.py`
- `tests/test_chunker.py`

### Indexing Behavior

- `BaseIndexer.ingest_documents(...)` accepts `list[DocumentChunkSchema]` or a new explicit indexing batch schema.
- `QdrantIndexer` writes chunk payloads using:
  - `file_id`
  - `section_id`
  - `asset_id`
  - `page_numbers`
  - `scopes`
  - `tags`
  - `chunk_metadata`
- Qdrant payload enum replaces `FILE_NAME` with `FILE_ID`.
- Legacy parent/child terminology is removed from new indexer interfaces.

### Retrieval & Generation Behavior

- Retrieval filters and generation tools accept `file_id` instead of `file_name`.
- Page lookup filters by `file_id` and `page_numbers`.
- Table lookup filters by `file_id` and `table_id` if table metadata remains supported.
- Existing references to `ParentChunk` are removed or isolated in `legacy_schema.py`.

### Postgres/Qdrant Relationship

- Postgres remains the system of record.
- Qdrant stores only searchable chunk payloads and enough metadata to map results back to Postgres/domain records.
- Preferred final design: Qdrant returns chunk hits with `section_id`; retriever hydrates full `DocumentSection.text` from Postgres.
- Temporary fallback: Qdrant may store enough section text payload to keep generation working while hydration is implemented.

### Phase 3 Tests

Create or update tests for:

- Indexer accepts new chunk schemas and writes `file_id` payloads.
- Retrieval exact filters reject legacy `file_name`.
- Page lookup uses `file_id`.
- Generation tools expose `file_id` in tool argument schemas.
- Qdrant hit hydration can recover section context through `section_id`.
- No production code imports from `rag_core.legacy_schema` after migration.
- Compatibility exports are removed or reduced after all call sites migrate.

## Implementation Order

1. Implement Phase 1 completely and commit it.
2. Run `uv run pytest` and scoped Ruff checks.
3. Review lifecycle behavior against Postgres before moving to content writes.
4. Implement Phase 2 in a separate commit.
5. Run the full test suite and verify Qdrant tests still pass.
6. Implement Phase 3 in a separate commit.
7. Run migration, ingestion, retrieval, and generation tests together.

## Explicit Non-Goals By Phase

### Phase 1 Non-Goals

- No PDF parsing.
- No markdown chunking.
- No table summarization.
- No Qdrant writes.
- No generation or retrieval changes.
- No schema package rename yet.

### Phase 2 Non-Goals

- No Qdrant payload migration.
- No generation tool changes from `file_name` to `file_id`.
- No removal of legacy schema compatibility exports.

### Phase 3 Non-Goals

- No new file lifecycle semantics.
- No new Postgres table design unless a Phase 2 implementation bug requires a migration.
- No broad prompt redesign.

## Assumptions

- `test_speed.py` remains untracked and out of scope.
- The current committed Postgres SQLAlchemy models are the source of truth.
- `file_id` is the canonical identifier for all new code.
- Existing Qdrant generation remains operational until Phase 3.
- `rag_core/pipelines/ingestion.py` remains a compatibility entrypoint unless explicitly removed later.
- Advisory lock name is exactly `ingestion_scanner`.
