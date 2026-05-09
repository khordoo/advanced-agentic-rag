# Ingestion

The ingestion input root is `data/source_documents/`.

## Pipeline Entrypoint

Use the canonical ingestion module for end-to-end ingestion:

```bash
uv run python -m rag_core.ingestion.pipeline
```

Discovery queues source files in Postgres. The ingestion worker then claims
queued assets, processes them, and exits when no claimable work remains.

## Worker Job Limits

By default, `max_jobs` is `None`. That means the worker runs until the queue is
empty and processes all available files.

`INGESTION_DEFAULT_MAX_JOBS` is an optional limit for one ingestion run. Keep it
commented out for the default drain-the-queue behavior:

```bash
# Optional limit for one ingestion run. If left commented out, max_jobs
# defaults to None and the worker runs until the queue is empty.
# INGESTION_DEFAULT_MAX_JOBS=100
```

Use this setting for local testing, smoke tests, canary runs, or bounded
orchestrator tasks. It is not the primary fix for memory leaks; if long ingestion
runs leak memory, investigate the processor or indexer lifecycle directly.

Multiple workers can process the same queue concurrently because each worker
claims assets through database leases.

## Metadata Extension Hooks

Metadata extraction lives in `rag_core/ingestion/metadata.py`.

The pipeline calls `extract_metadata_from_uri(source_uri)`, which delegates to these intentional extension hooks:
- `_extract_scopes(path_obj)`
- `_extract_tags(path_obj)`
- `_extract_extended_metadata(path_obj)`

These hooks are meant to be overridden for project-specific rules (for example: enterprise scope policies, custom tags, or additional JSON metadata).

Recommended contract:
- Keep hook function signatures unchanged.
- Keep return types stable:
  - `list[str]` for scopes and tags
  - `dict[str, object]` for extended metadata
- Keep `extract_metadata_from_uri(...)` as the stable entrypoint used by ingestion.

## Supported Input Files

- `.pdf`
- `.md`

## Folder-to-Tag Mapping

The immediate parent folder name is mapped to the document tag.

Examples:
- `data/source_documents/finance/report.pdf` -> `finance`
- `data/source_documents/hr/policy.md` -> `hr`

## Files and Folders Ignored by Ingestion

Ignored folders (not traversed):
- `.DS_Store`
- `__pycache__`
- `.git`

Ignored files:
- `README.md`
- `LICENSE`
- `.gitignore`

This is why guidance docs should live in `docs/` or as `README.md` files inside `data/source_documents/`.
