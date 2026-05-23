from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_docker_services_override_qdrant_url_for_container_dns() -> None:
    compose = yaml.safe_load((ROOT / "docker" / "docker-compose.yaml").read_text())
    services = compose["services"]

    api_env = services["rag_api"]["environment"]
    worker_env = services["ingestion-worker"]["environment"]

    assert "QDRANT_URL=http://qdrant:6333" in api_env
    assert worker_env["QDRANT_URL"] == "http://qdrant:6333"
    assert services["rag_api"]["depends_on"]["qdrant"]["condition"] == "service_started"


def test_env_example_documents_qdrant_host_vs_docker_usage() -> None:
    env_example = (ROOT / ".env.example").read_text()

    assert "QDRANT_URL=http://localhost:6333" in env_example
    assert "Docker compose overrides QDRANT_URL" in env_example
    assert "INGESTION_INDEXER_BACKEND=qdrant" in env_example
    assert "RETRIEVAL_BACKEND=qdrant" in env_example
    assert "fresh ingestion after switching backends" in env_example

