from __future__ import annotations

import pytest

prefect = pytest.importorskip("prefect")

from orchestration.ingestion import tasks  # noqa: E402


def test_end_to_end_task_raises_when_core_pipeline_reports_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Result:
        succeeded = False
        processed_jobs = 0
        discovery = None

    monkeypatch.setattr(tasks, "run_end_to_end_pipeline", lambda **_: _Result())

    with pytest.raises(RuntimeError, match="reported failure"):
        tasks.end_to_end_ingestion_task.fn(max_jobs=None)


def test_end_to_end_task_raises_migration_guidance_for_schema_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(**_):
        raise RuntimeError('relation "ingestion_runs" does not exist')

    monkeypatch.setattr(tasks, "run_end_to_end_pipeline", _boom)

    with pytest.raises(RuntimeError, match="make migrate-up"):
        tasks.end_to_end_ingestion_task.fn(max_jobs=None)


def test_ingestion_task_removes_overwrite_and_uses_max_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int | None] = []

    def _run_ingestion_pipeline(*, max_jobs: int | None = None):
        calls.append(max_jobs)
        return 3

    monkeypatch.setattr(tasks, "run_ingestion_pipeline", _run_ingestion_pipeline)

    assert tasks.ingestion_task.fn(max_jobs=7) == 3
    assert calls == [7]


def test_run_database_migrations_task_calls_alembic_upgrade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[tuple[object, str]] = []

    def _upgrade(config, revision):
        called.append((config, revision))

    monkeypatch.setattr(tasks.command, "upgrade", _upgrade)
    tasks.run_database_migrations_task.fn()
    assert len(called) == 1
    assert called[0][1] == "head"


def test_reset_vector_backend_task_resets_collections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"calls": 0}

    class _Indexer:
        def reset_collections(self) -> None:
            state["calls"] += 1

    class _Worker:
        indexer = _Indexer()

    monkeypatch.setattr(
        tasks.SourceIngestionWorkerService,
        "default_production",
        lambda: _Worker(),
    )
    tasks.reset_vector_backend_task.fn()
    assert state["calls"] == 1


def test_reset_ingestion_registries_task_truncates_tables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = {"sql": None, "commits": 0}

    class _Session:
        def execute(self, statement) -> None:
            seen["sql"] = str(statement)

        def commit(self) -> None:
            seen["commits"] += 1

    class _Database:
        def get_session(self):
            session = _Session()

            class _Ctx:
                def __enter__(self):
                    return session

                def __exit__(self, exc_type, exc, tb):
                    return False

            return _Ctx()

    monkeypatch.setattr(tasks, "get_database", lambda: _Database())
    tasks.reset_ingestion_registries_task.fn()

    table_names = f"{tasks.SourceAsset.__tablename__}, {tasks.IngestionRun.__tablename__}"
    assert f"TRUNCATE TABLE {table_names} RESTART IDENTITY CASCADE" in seen["sql"]
    assert seen["commits"] == 1
