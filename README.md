# Advanced RAG Pipeline

Production-oriented Retrieval-Augmented Generation pipeline with hierarchical chunking, reranking, hybrid retrieval, and evaluation tooling.

## Start Here

Use this file as the landing page, then go to detailed guides in [`docs/`](docs/README.md).

- [Documentation Index](docs/README.md)
- [Getting Started](docs/getting-started/README.md)
- [Configuration](docs/configuration/README.md)
- [Retrieval RRF Semantics](docs/configuration/retrieval-rrf.md)
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

# Start local infrastructure and services
docker compose --project-directory . -f docker/docker-compose.yaml up -d

# Default worker image includes PDF Docling processing.
# If you also need local reranking in the worker image:
# INGESTION_WORKER_UV_SYNC_EXTRAS="--extra ingestion --extra doc-processing --extra rerank" \
# docker compose --project-directory . -f docker/docker-compose.yaml up -d --build

# Open Prefect dashboard
# http://localhost:4200

# Run end-to-end ingestion flow
uv run --extra ingestion python -m orchestration.ingestion.flows end_to_end

# Register a clickable deployment in Prefect UI (one-time or after flow changes)
uv run --extra ingestion python -m orchestration.ingestion.flows end_to_end \
  --register-deployment --deployment-name local-ingestion-runner --work-pool ingestion-pool

# Equivalent Makefile shortcut
make register-ingestion-deployment

# Register both safe ingestion + destructive factory-reset deployments
make register-ingestion-deployments

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

## Prefect UI Trigger

Use Deployments in Prefect UI for one-click ingestion runs:
1. Start stack (`docker compose --project-directory . -f docker/docker-compose.yaml up -d`).
2. Register deployments via the command above (`make register-ingestion-deployments`).
3. Open `http://localhost:4200`, go to Deployments, then click Quick Run on the safe ingestion deployment.

For destructive reset operations, run the separate `*-factory-reset` deployment and set `confirm_destructive_wipe=true`.

If runs fail on first start due to missing schema, run:

```bash
make migrate-up
```

If Docker builds fail with `No space left on device` while extracting NVIDIA/CUDA wheels, use the default lightweight extras (already in `.env.example`) or clean disk first:

```bash
docker system prune -af --volumes
uv cache clean
```

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
├── app/                    # API transport layer
├── orchestration/          # Prefect flows/worker runtime
├── rag_core/
├── infra/                  # Future IaC
├── docs/
├── data/source_documents/
├── examples/
├── evaluation/
├── docker/
└── tests/
```

## License

MIT
