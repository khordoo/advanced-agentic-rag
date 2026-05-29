# Ingestion

The ingestion input root is `data/source_documents/`.

## Pipeline Entrypoint

Use the canonical ingestion module for end-to-end ingestion:

```bash
uv run python -m rag_core.ingestion.pipeline
```

For orchestration runtime (Prefect), use:

```bash
uv run --extra ingestion python -m orchestration.ingestion.flows end_to_end
```

If you ingest PDFs via Docling, install optional processing deps:

```bash
uv sync --extra ingestion --extra doc-processing
```

Register a deployment so the flow appears as a clickable run target in Prefect UI:

```bash
uv run --extra ingestion python -m orchestration.ingestion.flows end_to_end \
  --register-deployment --deployment-name local-ingestion-runner --work-pool ingestion-pool
```

After registration, open `http://localhost:4200`, navigate to Deployments, and use Quick Run.

To register both the safe ingestion deployment and the destructive admin reset deployment:

```bash
uv run --extra ingestion python -m orchestration.ingestion.flows end_to_end \
  --register-all-deployments --deployment-name local-ingestion-runner --work-pool ingestion-pool
```

The `*-factory-reset` deployment requires `confirm_destructive_wipe=true` and wipes vectors plus ingestion registry state.

Discovery queues source files in Postgres. The ingestion worker then claims
queued assets, processes them, and exits when no claimable work remains.

By default, discovery fails when zero eligible files are found. This protects
against misconfigured source paths (for example, missing Docker mounts). If you
need empty-source runs intentionally, set:

```bash
INGESTION_ALLOW_EMPTY_DISCOVERY=true
```

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

## Troubleshooting Logs

- Prefect UI run/task logs remain available in the dashboard.
- Worker application logs are also mirrored to container stdout:

```bash
docker logs rag-worker -f
```

- Per-asset failures are persisted in `source_assets.error_message` with stage and path context.

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
