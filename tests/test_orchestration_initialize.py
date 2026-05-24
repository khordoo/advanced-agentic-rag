from __future__ import annotations

import asyncio
from types import SimpleNamespace

from orchestration import initialize
from orchestration.config import PrefectConfig


def test_wait_for_server_uses_configured_api_url(
    monkeypatch,
) -> None:
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        initialize,
        "config",
        PrefectConfig(
            api_url="http://prefect.example/api",
            initial_backoff=0.01,
            max_wait_seconds=1,
            _env_file=None,
        ),
    )

    def _get(url: str, timeout: int):
        seen["url"] = url
        seen["timeout"] = timeout
        return SimpleNamespace(status_code=200)

    monkeypatch.setattr(initialize.httpx, "get", _get)

    initialize.wait_for_server()

    assert seen == {"url": "http://prefect.example/api/health", "timeout": 5}


def test_ensure_work_pool_uses_configured_pool_when_it_exists(
    monkeypatch,
) -> None:
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        initialize,
        "config",
        PrefectConfig(work_pool_name="custom-pool", work_pool_type="process", _env_file=None),
    )

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def read_work_pool(self, name: str):
            seen["pool_name"] = name
            return SimpleNamespace(name=name, type="process")

    monkeypatch.setattr(initialize, "get_client", lambda: _Client())

    asyncio.run(initialize.ensure_work_pool())

    assert seen == {"pool_name": "custom-pool"}


def test_register_deployments_uses_configured_deployment_and_pool(
    monkeypatch,
) -> None:
    records: list[dict[str, object]] = []
    monkeypatch.setattr(
        initialize,
        "config",
        PrefectConfig(
            work_pool_name="custom-pool",
            deployment_name="custom-deployment",
            _env_file=None,
        ),
    )

    class _FlowFn:
        __name__ = "example_flow"

    class _Flow:
        name = "example-flow"
        fn = _FlowFn()

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def create_flow_from_name(self, name: str):
            return f"{name}-id"

        async def create_deployment(self, **kwargs):
            records.append(kwargs)
            return "deployment-id"

    monkeypatch.setattr(
        initialize,
        "_DEPLOYMENT_SPECS",
        [(_Flow(), "", ["safe"], "description")],
    )
    monkeypatch.setattr(initialize, "get_client", lambda: _Client())

    asyncio.run(initialize.register_deployments())

    assert records == [
        {
            "flow_id": "example-flow-id",
            "name": "custom-deployment",
            "work_pool_name": "custom-pool",
            "tags": ["safe"],
            "description": "description",
            "entrypoint": "orchestration/ingestion/flows.py:example_flow",
            "path": "/app",
        }
    ]
