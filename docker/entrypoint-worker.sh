#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# RAG Worker Container Entrypoint
# ──────────────────────────────────────────────────────────────────────────────
# This script runs inside the worker container on every boot.
# It performs two phases:
#   1. Pre-flight: idempotent infrastructure provisioning (work pool + deployments)
#   2. Worker:     hand off PID 1 to the Prefect worker process via exec
#
# Because this lives inside the image (not docker-compose.yaml), the container
# is self-contained and portable to Kubernetes, ECS, or any other runtime.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

POOL_NAME="${PREFECT_WORK_POOL:-rag-work-pool}"
WORKER_LIMIT="${PREFECT_WORKER_LIMIT:-2}"

echo "🚀 Running pre-flight orchestration setup …"
.venv/bin/python -m orchestration.initialize

echo "🐝 Starting Prefect worker (pool=${POOL_NAME}, limit=${WORKER_LIMIT}) …"
exec .venv/bin/prefect worker start \
    --pool "${POOL_NAME}" \
    --type process \
    --limit "${WORKER_LIMIT}"
