# Advanced RAG Pipeline

Production-oriented Retrieval-Augmented Generation pipeline with hierarchical chunking, reranking, hybrid retrieval, and evaluation tooling.

## Start Here

Use this file as the landing page, then go to detailed guides in [`docs/`](docs/README.md).

- [Documentation Index](docs/README.md)
- [Getting Started](docs/getting-started/README.md)
- [Configuration](docs/configuration/README.md)
- [Generation Model Settings](docs/configuration/generation-models.md)
- [Ingestion Rules](docs/ingestion/README.md)
- [Agents & Prompt Profiles](docs/agents/README.md)

## Quick Start

```bash
# Clone and install
git clone <your-repo-url> && cd advanced-rag
uv sync --extra dev

# Configure
cp .env.example .env

# Ingest source files from data/source_documents/
uv run python -m rag_core.ingestion.pipeline

# Start CLI
uv run rag_core/cli.py
```

## 🚨 Point of No Return: Changing Embedding Models

> [!CAUTION]
> **Embedding models are immutable for existing indices.**
> 
> Vectors are mathematically tied to the specific model that generated them. If you change your `EMBEDDING_MODEL_NAME` or `EMBEDDING_PROVIDER` in your settings:
> 
> - **Vector Mismatch**: If the new model has a different dimension, all existing database queries and insertions will fail.
> - **Semantic Drift**: Even if the dimensions match, the new model cannot "understand" vectors created by the old model. Your search results will become random noise.
> 
> **Requirement**: Changing your embedding configuration requires a full re-ingestion. For a one-off rebuild, run the ingestion entrypoint with `INGESTION_OVERWRITE=true`:
>
> ```bash
> INGESTION_OVERWRITE=true uv run python -m rag_core.ingestion.pipeline
> ```
>
> This deletes existing vectors for the selected backend before rebuilding them. Do not keep `INGESTION_OVERWRITE=true` in committed `.env` files or long-lived deployment config; run future ingestion with overwrite disabled to avoid accidental data loss.

Provider names are intentionally strict. If you add a new embedding backend,
update the enum and factory wiring described in
[`docs/configuration/README.md`](docs/configuration/README.md#adding-an-embedding-provider).

## Database Migrations

Use `make` shortcuts for Alembic commands (config is wired to `db/alembic.ini`).

```bash
# Apply latest migration
make migrate-up

# Upgrade/downgrade to a specific revision (or relative step like -1 / +1)
make migrate-up REV=<revision_or_step>
make migrate-down REV=<revision_or_step>

# Create a new autogenerate migration
make migrate-new MSG="describe schema change"

# Show migration history
make migrate-status
```

## Ingestion Worker Behavior

The canonical ingestion entrypoint is `rag_core.ingestion.pipeline`. Discovery
queues source files in Postgres, then the ingestion worker claims queued assets
and processes them until no claimable work remains.

By default, `max_jobs` is `None`, which means the worker drains the queue and
processes all available files. Multiple workers can run at the same time because
asset claims are coordinated through database leases.

`INGESTION_DEFAULT_MAX_JOBS` is an optional limit for local testing, smoke tests,
canary runs, or bounded orchestrator tasks. Leave it commented out for the
default drain-the-queue behavior.

## Folder Conventions

- `docs/` contains project guides and operational documentation.
- `data/source_documents/` contains ingestable source files (`.pdf`, `.md`).
- Folder-tag behavior and ignore rules for ingestion are documented in:
  - [`docs/ingestion/README.md`](docs/ingestion/README.md)
  - [`data/source_documents/README.md`](data/source_documents/README.md)
- Metadata extraction extension hooks (scopes/tags/extended metadata) are also documented in:
  - [`docs/ingestion/README.md`](docs/ingestion/README.md)

## Project Structure

```text
advanced-rag/
├── rag_core/
├── docs/
├── data/source_documents/
├── examples/
├── evaluation/
├── docker/
└── tests/
```

## License

MIT
