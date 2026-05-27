from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_docker_services_pass_through_docker_first_urls() -> None:
    compose = yaml.safe_load((ROOT / "docker" / "docker-compose.yaml").read_text())
    services = compose["services"]

    api_env = services["rag_api"]["environment"]
    worker_env = services["ingestion-worker"]["environment"]
    web_env = services["web_ui"]["environment"]
    api_build_args = services["rag_api"]["build"]["args"]
    worker_build_args = services["ingestion-worker"]["build"]["args"]

    assert "POSTGRES_HOST=${POSTGRES_HOST:-core-db}" in api_env
    assert "QDRANT_URL=${QDRANT_URL:-http://qdrant:6333}" in api_env
    assert (
        "OPEN_AI_COMPATIBLE_BASE_URL="
        "${OPEN_AI_COMPATIBLE_BASE_URL:-http://host.docker.internal:1234/v1}"
        in api_env
    )
    assert "OLLAMA_BASE_URL=${OLLAMA_BASE_URL:-http://host.docker.internal:11434}" in api_env
    assert worker_env["POSTGRES_HOST"] == "${POSTGRES_HOST:-core-db}"
    assert worker_env["QDRANT_URL"] == "${QDRANT_URL:-http://qdrant:6333}"
    assert worker_env["PREFECT_API_URL"] == "${PREFECT_API_URL:-http://prefect-server:4200/api}"
    assert worker_env["PREFECT_LOGGING_EXTRA_LOGGERS"] == "${PREFECT_LOGGING_EXTRA_LOGGERS:-rag_core}"
    assert "OPENAI_API_BASE_URL=${OPENAI_API_BASE_URL:-http://rag_api:8000/v1}" in web_env
    assert "OPENAI_API_KEY=${OPENAI_API_KEY:-dummy_key}" in web_env
    assert services["rag_api"]["depends_on"]["qdrant"]["condition"] == "service_started"
    assert api_build_args["EMBEDDING_MODEL_NAME"] == "${EMBEDDING_MODEL_NAME:-BAAI/bge-small-en-v1.5}"
    assert (
        api_build_args["QDRANT_SPARSE_MODEL"]
        == "${QDRANT_SPARSE_MODEL:-Qdrant/bm42-all-minilm-l6-v2-attentions}"
    )
    assert api_build_args["RERANK_MODEL_NAME"] == "${RERANK_MODEL_NAME:-BAAI/bge-reranker-v2-m3}"
    assert api_build_args["HF_TOKEN"] == "${HF_TOKEN:-}"
    assert worker_build_args["EMBEDDING_MODEL_NAME"] == "${EMBEDDING_MODEL_NAME:-BAAI/bge-small-en-v1.5}"
    assert (
        worker_build_args["QDRANT_SPARSE_MODEL"]
        == "${QDRANT_SPARSE_MODEL:-Qdrant/bm42-all-minilm-l6-v2-attentions}"
    )
    assert worker_build_args["HF_TOKEN"] == "${HF_TOKEN:-}"
    assert "RERANK_MODEL_NAME" not in worker_build_args


def test_env_example_defaults_are_docker_compose_first() -> None:
    env_example = (ROOT / ".env.example").read_text()

    assert "OPEN_AI_COMPATIBLE_BASE_URL='http://host.docker.internal:1234/v1'" in env_example
    assert "OLLAMA_BASE_URL='http://host.docker.internal:11434'" in env_example
    assert "QDRANT_URL=http://qdrant:6333" in env_example
    assert "POSTGRES_HOST=core-db" in env_example
    assert "PREFECT_API_URL=http://prefect-server:4200/api" in env_example
    assert "PREFECT_INGESTION_WORK_POOL=rag-work-pool" in env_example
    assert "PREFECT_DEPLOYMENT_NAME=ingest-documents" in env_example
    assert "PREFECT_INIT_MAX_WAIT=120" in env_example
    assert "PREFECT_LOGGING_EXTRA_LOGGERS=rag_core" in env_example
    assert "OPENAI_API_BASE_URL=http://rag_api:8000/v1" in env_example
    assert "localhost or another reachable host" in env_example
    assert "Makefile host-run targets already do" in env_example
    assert "Docker compose overrides QDRANT_URL" not in env_example
    assert "INGESTION_INDEXER_BACKEND=qdrant" in env_example
    assert "RETRIEVAL_BACKEND=qdrant" in env_example
    assert "QDRANT_SPARSE_MODEL=Qdrant/bm42-all-minilm-l6-v2-attentions" in env_example
    assert "fresh ingestion after switching backends" in env_example
    assert 'HF_TOKEN=""' in env_example
    assert "INGESTION_WORKER_UV_SYNC_EXTRAS=--extra ingestion --extra doc-processing --extra rerank" not in env_example


def test_makefile_host_targets_translate_docker_service_names() -> None:
    makefile = (ROOT / "Makefile").read_text()

    assert "HOST_POSTGRES_HOST?=localhost" in makefile
    assert "HOST_QDRANT_URL?=http://localhost:6333" in makefile
    assert "HOST_PREFECT_API_URL?=http://localhost:4200/api" in makefile
    assert "HOST_OTEL_EXPORTER_OTLP_TRACES_ENDPOINT?=http://localhost:6006/v1/traces" in makefile
    assert "HOST_ENV:=POSTGRES_HOST=$(HOST_POSTGRES_HOST)" in makefile
    assert "$(HOST_ENV) $(ALEMBIC_BASE) -c $(ALEMBIC_CONFIG)" in makefile
    assert "ingestion-worker run --no-sync python -m orchestration.ingestion.flows end_to_end" in makefile
