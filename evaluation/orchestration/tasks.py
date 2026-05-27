"""Prefect @task wrappers for evaluation pipeline stages.

Each task starts with ``setup_logging()`` + ``_get_logger()`` per the
ingestion convention.  One task per pipeline stage for clear Prefect
UI observability.
"""

from __future__ import annotations

import logging

from prefect import get_run_logger, task
from prefect.cache_policies import NO_CACHE
from prefect.exceptions import MissingContextError

from rag_core.core.config import setup_logging

logger = logging.getLogger(__name__)


def _get_logger() -> logging.Logger:
    """Return the Prefect run logger when available, else stdlib."""
    try:
        return get_run_logger()
    except MissingContextError:
        return logging.getLogger(__name__)


@task(
    name="eval-resolve-dataset",
    retries=1,
    retry_delay_seconds=5,
)
def resolve_dataset_task(eval_config) -> str:
    """Upload / verify the golden dataset in Phoenix and return its handle."""
    setup_logging()
    log = _get_logger()
    log.info("Resolving evaluation dataset '%s'", eval_config.dataset_name)

    from evaluation.datasets.loaders import ensure_dataset_uploaded  # lazy

    dataset_handle = ensure_dataset_uploaded(eval_config)
    log.info("Dataset resolved: %s", dataset_handle)
    return dataset_handle


@task(name="eval-run-experiment", retries=0, cache_policy=NO_CACHE)
def run_experiment_task(
    dataset,
    task_fn,
    evaluators,
    experiment_name: str,
    concurrency: int,
    project_name: str,
) -> object:
    """Run a single Phoenix experiment.

    ``retries=0`` — re-runs would create duplicate Experiments.
    """
    setup_logging()
    log = _get_logger()
    log.info(
        "Running experiment '%s' (concurrency=%d, project=%s)",
        experiment_name,
        concurrency,
        project_name,
    )

    from evaluation.phoenix_bootstrap import get_phoenix_client  # lazy

    client = get_phoenix_client()
    experiment = client.experiments.run_experiment(
        dataset=dataset,
        task=task_fn,
        evaluators=evaluators,
        experiment_name=experiment_name,
        experiment_metadata={"project_name": project_name},
    )

    log.info("Experiment '%s' completed", experiment_name)
    return experiment


@task(name="eval-apply-thresholds")
def apply_thresholds_task(experiment, eval_config, aggregates: dict[str, float]):
    """Apply pass/fail thresholds to experiment aggregates."""
    setup_logging()
    log = _get_logger()
    log.info("Applying thresholds to experiment results")

    from evaluation.evaluators.thresholds import evaluate_thresholds  # lazy

    report = evaluate_thresholds(aggregates, eval_config)
    log.info("Threshold evaluation: %s", report.summary())
    return report


@task(name="eval-emit-artifact")
def emit_artifact_task(
    experiment, pass_fail, eval_config, aggregates: dict[str, float] | None = None
):
    """Emit Prefect markdown + optional CSV artifacts."""
    setup_logging()
    log = _get_logger()
    log.info("Emitting evaluation artifacts")

    from evaluation.reporting.artifacts import emit_csv_artifact  # lazy
    from evaluation.reporting.artifacts import emit_experiment_artifact  # lazy

    emit_experiment_artifact(experiment, pass_fail, eval_config, aggregates=aggregates)
    emit_csv_artifact(experiment, eval_config)
    log.info("Artifacts emitted ✓")
