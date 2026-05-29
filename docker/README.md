# 🐳 Docker Stack

## Quick Start

```bash
# Start the full stack (everything is auto-provisioned)
docker compose --project-directory . -f docker/docker-compose.yaml up -d

# Check services are healthy
docker compose --project-directory . -f docker/docker-compose.yaml ps
```

Default worker and API images include runtime code and dependencies only. The
`model-warmer` service downloads the dense, sparse, and `BAAI/bge-reranker-base`
FastEmbed models into a shared Docker volume before the API and worker start.

## Services

| Service | Port | Purpose |
|---|---|---|
| **Prefect Server** | `4200` | Orchestration UI/API |
| **RAG API** | `8000` | OpenAI-compatible RAG API |
| **Web UI** | `3000` | Chat interface |
| **Qdrant** | `6333` (REST), `6334` (gRPC) | Vector DB with hybrid search (BM25 + vector) |
| **Phoenix** | `6006` | OpenTelemetry/OpenInference tracing UI |
| **PostgreSQL (core-db)** | `5432` | Shared DB host (RAG + Prefect metadata) |
| **RAG Worker** | internal | Prefect process worker for ingestion and evaluation flows |
| **Model Warmer** | internal | Preloads FastEmbed models into the shared cache volume |

After startup, open:
- Prefect dashboard: `http://localhost:4200`
- RAG API health: `http://localhost:8000/ping`
- Web UI: `http://localhost:3000`
- Qdrant dashboard: `http://localhost:6333/dashboard`
- Phoenix: `http://localhost:6006`

## Auto-Provisioning (Zero Manual Steps)

The RAG worker container is fully self-bootstrapping. On every startup, its
entrypoint script (`docker/entrypoint-worker.sh`) runs the pre-flight initialization
module (`orchestration/initialize.py`), which:

1. **Waits for the Prefect API** to become healthy (exponential backoff).
2. **Creates the `rag-work-pool` work pool** (type: `process`) if it doesn't exist.
3. **Registers all canonical deployments** to the Prefect server:
   - `ingest-documents` — discover, process, and index source documents for retrieval.
   - `reset-document-index` — destructive reset of indexed documents and ingestion state.
   - `evaluate-ground-truth` — evaluate RAG answers against the ground truth dataset.
   - `evaluate-traces` — evaluate recent production traces with reference-free judges.
   - `generate-synthetic-qa` — generate synthetic QA examples from indexed documents.

All operations are **idempotent** — restarting the stack or re-creating containers
is always safe and never produces duplicate resources.

> **Database migrations** are applied automatically as a fail-fast preflight task
> inside the ingestion flow itself (`run_database_migrations_task`), ensuring the
> relational schema is always current before any document processing begins.

## Startup Modes

The stack supports two startup modes via Makefile:

| Command | Services | Use Case |
|---------|----------|----------|
| `make up-core` | DB, Qdrant, Phoenix, Prefect, Worker, API | Headless dev: ingestion, eval, CLI |
| `make up` | Everything above + Open-WebUI | Full stack with chat UI |

## Run Ingestion Flow

Trigger ingestion from Prefect UI (recommended):

1. Open `http://localhost:4200`
2. Go to **Deployments**
3. Open `ingest-documents`
4. Click **Quick Run** for immediate ingestion, or create a **Schedule** from the same deployment

If `ingest-documents` is not visible in Deployments:

```bash
make register-ingestion-deployments
```

Host-side manual trigger (optional):

```bash
make run-ingestion-flow
```

The `.env` file is Docker Compose-first. Makefile host-run targets translate
Docker service hostnames to host-reachable defaults through `HOST_*` variables
such as `HOST_POSTGRES_HOST=localhost` and `HOST_QDRANT_URL=http://localhost:6333`.

## Data Persistence

All data is stored in Docker volumes:
- `qdrant_data` — Vector DB storage
- `postgres_data` — PostgreSQL data directory
- `open-webui-data` — Open WebUI data
- `phoenix_data` — Phoenix trace storage

To reset the stack and wipe all Docker volumes, including Phoenix traces:
```bash
make reset
```

If image builds fail due to disk pressure (`No space left on device`), clean caches and old layers:

```bash
docker system prune -af --volumes
uv cache clean
```
