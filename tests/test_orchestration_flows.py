from __future__ import annotations

import pytest

prefect = pytest.importorskip("prefect")

from orchestration.ingestion import flows


def test_end_to_end_flow_runs_migrations_before_ingestion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        flows,
        "run_database_migrations_task",
        lambda: calls.append("migrate"),
    )
    monkeypatch.setattr(
        flows,
        "end_to_end_ingestion_task",
        lambda **_: calls.append("ingest") or {"succeeded": True},
    )

    flows.end_to_end_flow(max_jobs=1)
    assert calls == ["migrate", "ingest"]


def test_factory_reset_flow_requires_confirmation() -> None:
    with pytest.raises(ValueError, match="confirm_destructive_wipe=true"):
        flows.factory_reset_flow(confirm_destructive_wipe=False)


def test_factory_reset_flow_runs_both_reset_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    monkeypatch.setattr(flows, "reset_vector_backend_task", lambda: calls.append("vector"))
    monkeypatch.setattr(
        flows,
        "reset_ingestion_registries_task",
        lambda: calls.append("registry"),
    )

    flows.factory_reset_flow(confirm_destructive_wipe=True)
    assert calls == ["vector", "registry"]


def test_register_all_deployments_registers_safe_and_destructive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records: list[dict[str, object]] = []

    def _record(target_flow, deployment_name, work_pool, work_queue, tags, description):
        records.append(
            {
                "target_flow": target_flow,
                "deployment_name": deployment_name,
                "work_pool": work_pool,
                "work_queue": work_queue,
                "tags": tags,
                "description": description,
            }
        )

    monkeypatch.setattr(flows, "_register_deployment", _record)
    monkeypatch.setattr(
        flows,
        "_build_parser",
        lambda: _ParserStub(
            flow_name="end_to_end",
            max_jobs=None,
            register_deployment=False,
            register_all_deployments=True,
            deployment_name="local-ingestion-runner",
            work_pool="ingestion-pool",
            work_queue=None,
        ),
    )

    flows.main()
    assert len(records) == 2
    assert records[0]["target_flow"] is flows.end_to_end_flow
    assert records[1]["target_flow"] is flows.factory_reset_flow
    assert records[0]["deployment_name"] == "local-ingestion-runner"
    assert records[1]["deployment_name"] == "local-ingestion-runner-factory-reset"
    assert "safe" in records[0]["tags"]
    assert "dangerous" in records[1]["tags"]


class _ParserStub:
    def __init__(self, **kwargs) -> None:
        self._kwargs = kwargs

    def parse_args(self):
        return type("Args", (), self._kwargs)()
