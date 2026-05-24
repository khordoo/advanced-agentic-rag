from __future__ import annotations

import pytest
from pydantic import ValidationError

from orchestration.config import PrefectConfig


PREFECT_ENV_KEYS = (
    "PREFECT_API_URL",
    "PREFECT_INGESTION_WORK_POOL",
    "PREFECT_DEPLOYMENT_NAME",
    "PREFECT_INIT_MAX_WAIT",
)


@pytest.fixture(autouse=True)
def clear_prefect_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in PREFECT_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_prefect_config_defaults_match_local_runtime() -> None:
    config = PrefectConfig(_env_file=None)

    assert config.api_url == "http://localhost:4200/api"
    assert config.work_pool_name == "ingestion-pool"
    assert config.work_pool_type == "process"
    assert config.deployment_name == "local-ingestion-runner"
    assert config.max_wait_seconds == 120
    assert config.initial_backoff == 1.0
    assert config.max_backoff == 10.0


def test_prefect_config_reads_prefect_env_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PREFECT_API_URL", "http://prefect-server:4200/api")
    monkeypatch.setenv("PREFECT_INGESTION_WORK_POOL", "custom-pool")
    monkeypatch.setenv("PREFECT_DEPLOYMENT_NAME", "custom-deployment")
    monkeypatch.setenv("PREFECT_INIT_MAX_WAIT", "42")

    config = PrefectConfig(_env_file=None)

    assert config.api_url == "http://prefect-server:4200/api"
    assert config.work_pool_name == "custom-pool"
    assert config.deployment_name == "custom-deployment"
    assert config.max_wait_seconds == 42


def test_prefect_config_rejects_invalid_wait_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PREFECT_INIT_MAX_WAIT", "abc")

    with pytest.raises(ValidationError, match="PREFECT_INIT_MAX_WAIT"):
        PrefectConfig(_env_file=None)
