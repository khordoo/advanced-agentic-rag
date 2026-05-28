# Advanced RAG Pipeline

Production-oriented Retrieval-Augmented Generation pipeline with hierarchical chunking, reranking, hybrid retrieval, and automated evaluation via Phoenix Experiments + RAGAS.

## Start Here

Use this file as the landing page, then go to detailed guides in [`docs/`](docs/README.md).

- [Documentation Index](docs/README.md)
- [Getting Started](docs/getting-started/README.md)
- [Configuration](docs/configuration/README.md)
- [Retrieval RRF Semantics](docs/configuration/retrieval-rrf.md)
- [Generation Model Settings](docs/configuration/generation-models.md)
- [Ingestion Rules](docs/ingestion/README.md)
- [Agents & Prompt Profiles](docs/agents/README.md)
- [Evaluation Pipeline](#evaluation-pipeline) ← new

## Quick Start

```bash
# Clone and install
git clone <your-repo-url> && cd advanced-rag
uv sync --extra dev

# Configure
cp .env.example .env
# Optional: Set HF_TOKEN in .env if you use gated models or hit rate limits.
# For standard use with default models, you can leave it blank.

# Start local infrastructure and services
docker compose --project-directory . -f docker/docker-compose.yaml up -d

# Open Prefect dashboard
# http://localhost:4200

# Register a clickable deployment in Prefect UI (one-time or after flow changes)
uv run --extra ingestion python -m orchestration.ingestion.flows end_to_end \
  --register-deployment --deployment-name ingest-documents --work-pool rag-work-pool

# Equivalent Makefile shortcut
make register-ingestion-deployment

# Register both document ingestion + destructive reset deployments
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
2. Open `http://localhost:4200`, go to **Deployments**.
3. Start ingestion from `ingest-documents` via **Quick Run** (or create a **Schedule** from that deployment page).
4. If `ingest-documents` is missing, register deployments with `make register-ingestion-deployments`, then refresh the Deployments page.

For destructive reset operations, run the separate `reset-document-index` deployment and set `confirm_destructive_wipe=true`.

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

## Evaluation Pipeline

The evaluation subsystem runs registry-backed RAGAS metrics plus deterministic project judges against a ground truth QA dataset, storing results as Phoenix Experiments with full trace correlation.

> [!IMPORTANT]
> **Evaluation is runtime-gated by default.** A fresh clone registers eval 
> deployments but only installs evaluation libraries (ragas, phoenix) when 
> `make eval` is run, keeping the base images lightweight.

### How It Works

```text
  ground_truth_dataset.csv ──→ Phoenix Dataset ("rag-ground-truth-v1")
                        │
                        ▼
      run_ground_truth_evaluation (Prefect @flow)
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
  resolve_dataset   run_experiment   apply_thresholds
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
     RAGAS Evaluators     Project Judges
  (Faithfulness, etc.)  (Citation format)
              │                   │
              └─────────┬─────────┘
                        ▼
              Prefect markdown artifact
           (Phoenix UI link + metrics table)
```

1. The ground truth QA set (`evaluation/datasets/ground_truth_dataset.csv`) is uploaded to Phoenix as a versioned Dataset (idempotent by content hash).
2. `run_ground_truth_evaluation` constructs a `RAGPipeline` once, runs it per example via `phoenix.experiments.run_experiment`, and evaluates with `GROUND_TRUTH_EVALUATORS`.
3. Results are stored as a Phoenix Experiment (visible in the Phoenix UI with per-example traces) and summarised as a Prefect markdown artifact.
4. Scores are logged to Phoenix and summarized as a Prefect artifact. The registry-backed CE flows do not fail only because a metric is low.

### Quick Start: Run Evaluation

```bash
# 1. Start infrastructure (includes Phoenix at http://localhost:6006)
make up

# 2. Upload ground truth dataset (idempotent)
docker compose --project-directory . --env-file .env -f docker/docker-compose.yaml \
  run --rm --entrypoint "uv" ingestion-worker \
  run --no-sync python -c \
  "from evaluation.datasets.loaders import ensure_dataset_uploaded; \
   from evaluation.config import get_eval_config; \
   print(ensure_dataset_uploaded(get_eval_config()))"

# 3. Run an experiment (one-shot)
make eval
# Or with a custom experiment name:
docker compose --project-directory . --env-file .env -f docker/docker-compose.yaml \
  run --rm --entrypoint "uv" ingestion-worker \
  run --no-sync python -c \
  "from evaluation.orchestration.prefect_flows import run_ground_truth_evaluation; \
   run_ground_truth_evaluation(experiment_name='my-first-eval')"

# Optional: run trace-based drift detection on demand
make eval-drift

# 4. Verify in Phoenix UI (http://localhost:6006):
#    - Project "advanced-rag-eval" exists
#    - Dataset "rag-ground-truth-v1" appears
#    - Experiment shows row count == ground truth row count
#    - Each row has evaluator scores + full RAG span tree

# 5. Register all Prefect deployments manually if needed:
make register-eval-deployment
```

`make eval` automatically detects missing dependencies and performs an idempotent
auto-upgrade of the ingestion-worker image (preserving ingestion and doc-processing
extras, then adding evaluation). It runs with `RERANK_ENABLED=false` by default
to keep CUDA and reranking out of the worker/eval path. For reranked evals, build
the worker separately with the `rerank` extra and run `make eval EVAL_RERANK_ENABLED=true`.

### Evaluation Modes

| Mode | Flow | Trigger | Ground Truth | Description |
|------|------|---------|-------------|-------------|
| **Ground truth** | `run_ground_truth_evaluation` | `EVAL_SCHEDULE_CRON` or manual | ✅ Required | Runs `GROUND_TRUTH_EVALUATORS` against `ground_truth_dataset.csv` |
| **Trace-based** | `run_nightly_drift_detection` | `EVAL_TRACE_SCHEDULE_CRON`, `make eval-drift`, or manual | ❌ Not needed | Runs `DRIFT_EVALUATORS` on failed-verification Phoenix spans |
| **Synthetic QA** | `synthetic_qa_flow` | Manual | N/A | Generates QA pairs from document store → Phoenix Dataset |

### Threshold Gates

Set `EVAL_THRESHOLDS` to a JSON dict to enforce minimum scores:

```bash
# Fail the flow if faithfulness < 0.9 or context_recall < 0.85
EVAL_THRESHOLDS='{"faithfulness": 0.9, "context_recall": 0.85}'
```

The registry-backed CE flows log scores to Phoenix and do not raise only because
a metric is low. `EVAL_THRESHOLDS` remains available for legacy threshold reports
and older evaluation paths.

### Post-Ingestion Hook

To automatically run evaluation after each ingestion:

```bash
EVAL_AUTO_RUN_AFTER_INGESTION=true
```

The hook is non-blocking — eval failures are logged but never break ingestion.

### EVAL_* Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EVAL_SCHEDULE_CRON` | _(empty)_ | Cron for scheduled ground truth eval |
| `EVAL_TRACE_SCHEDULE_CRON` | _(empty)_ | Cron for trace-based eval |
| `EVAL_AUTO_RUN_AFTER_INGESTION` | `false` | Run eval after ingestion |
| `EVAL_DATASET_NAME` | `rag-ground-truth-v1` | Phoenix Dataset name |
| `EVAL_DATASET_CSV_PATH` | `evaluation/datasets/ground_truth_dataset.csv` | Ground truth CSV path |
| `EVAL_PHOENIX_PROJECT` | `advanced-rag-eval` | Phoenix project (isolated from prod) |
| `EVAL_PHOENIX_ENDPOINT` | _(auto)_ | Override Phoenix endpoint |
| `EVAL_JUDGE_PROVIDER` | `aws` | LLM provider for judges |
| `EVAL_JUDGE_MODEL` | _(provider default)_ | Override judge model |
| `EVAL_TIMEOUT` | `60` | Maximum time to wait for a single metric evaluation |
| `EVAL_MAX_RETRIES` | `3` | Maximum retries for a failed metric evaluation |
| `EVAL_MAX_WORKERS` | `4` | Maximum concurrent evaluation tasks |
| `EVAL_MAX_EXAMPLES` | _(all)_ | Hard cap to bound judge cost |
| `EVAL_RAGAS_METRICS` | `faithfulness,answer_relevancy,...` | RAGAS metrics to run |
| `EVAL_PHOENIX_EVALUATORS` | `hallucination,qa_correctness,relevance` | Legacy Phoenix judges setting; not used by the registry-backed CE ground-truth flow |
| `EVAL_THRESHOLDS` | _(empty)_ | JSON dict of metric → min score |
| `EVAL_CSV_ARTIFACT_ENABLED` | `false` | Emit CSV Prefect artifact |
| `EVAL_TRACE_LOOKBACK_MINUTES` | `60` | Time window for trace eval |

> [!TIP]
> **OSS users without AWS:** Set `EVAL_JUDGE_PROVIDER=ollama` (or `lmstudio`) and `EVAL_JUDGE_MODEL=<your-model>` to use local judge models.

### Evaluation Package Structure

```text
evaluation/
├── config.py                    # EvalConfig(BaseSettings) + feature flags
├── phoenix_bootstrap.py         # Lazy phoenix.Client() factory
├── datasets/
│   ├── ground_truth_dataset.csv              # Golden QA set (hand-curated)
│   └── loaders.py               # CSV→Phoenix Dataset upload (idempotent)
├── generation/
│   └── synthetic.py             # Async QA generation → Phoenix Dataset
├── pipeline/
│   ├── task.py                  # run_rag_for_example (per-example isolation)
│   └── adapter.py               # Phoenix ↔ RAG ↔ RAGAS shape conversion
├── evaluators/
│   ├── ragas_metrics.py         # RAGAS metric → Phoenix evaluator factory
│   ├── custom_judges.py         # Deterministic project-specific judges
│   ├── phoenix_judges.py        # Legacy reference-free LLM-as-judge evaluators
│   └── thresholds.py            # Pass/fail policy (PassFailReport)
├── eval_registry.py             # GROUND_TRUTH_EVALUATORS + DRIFT_EVALUATORS
├── reporting/
│   └── artifacts.py             # Prefect markdown + CSV artifacts
└── orchestration/
    ├── prefect_flows.py         # Canonical CE flows
    ├── flows.py                 # Legacy wrappers + synthetic_qa
    ├── tasks.py                 # @task wrappers (one per stage)
    └── deployments.py           # Eval deployment specs
```

## Project Structure

```text
advanced-rag/
├── app/                    # API transport layer
├── orchestration/          # Prefect flows/worker runtime
├── rag_core/               # Core RAG engine
├── evaluation/             # Evaluation pipeline
│   ├── config.py           #   EvalConfig + feature flags
│   ├── datasets/           #   Golden QA sets + loaders
│   ├── evaluators/         #   RAGAS + Phoenix judge factories
│   ├── pipeline/           #   Per-example task + adapters
│   ├── reporting/          #   Prefect artifact emitters
│   └── orchestration/      #   Flows, tasks, deployment specs
├── docker/
├── docs/
├── data/source_documents/
├── tests/
└── examples/
```

## License

MIT
