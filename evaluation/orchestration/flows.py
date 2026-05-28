"""Prefect @flow definitions for the evaluation pipeline.

Top-level imports: stdlib + prefect + evaluation.config only.
All heavy deps (RagService, RagGraphFactory, evaluator factories, Phoenix client APIs)
are imported **lazily inside flow bodies** after runtime checks.

Pattern mirrors ``orchestration/ingestion/flows.py``.
"""

from __future__ import annotations

import logging

from prefect import flow, get_run_logger
from prefect.exceptions import MissingContextError

from evaluation.config import get_eval_config
from evaluation.orchestration.prefect_flows import (
    run_ground_truth_evaluation,
    run_nightly_drift_detection,
)

logger = logging.getLogger(__name__)


def _get_logger() -> logging.Logger:
    """Return the Prefect run logger when available, else stdlib."""
    try:
        return get_run_logger()
    except MissingContextError:
        return logging.getLogger(__name__)


def _assert_eval_runtime_ready() -> None:
    """Validate optional eval/runtime dependencies at flow execution time."""
    from evaluation.runtime import assert_eval_runtime_ready, EvaluationRuntimeError

    try:
        assert_eval_runtime_ready()
    except EvaluationRuntimeError:
        log = _get_logger()
        import sys

        sys.tracebacklimit = 0

        error_msg = (
            "\n\n" + "=" * 70 + "\n"
            "🚨 MISSING EVALUATION DEPENDENCIES\n\n"
            "The current worker image does not have evaluation libraries installed\n"
            "(ragas, phoenix-evals, etc.).\n\n"
            "To unlock evaluations, run this command in your terminal once:\n"
            "  make eval\n\n"
            "This will install the eval extras and trigger the pipeline.\n"
            + "=" * 70 + "\n"
        )

        log.error(error_msg)
        raise RuntimeError("Missing evaluation dependencies.") from None


# ---------------------------------------------------------------------------
# A/B) Compatibility wrappers for canonical CE flows
# ---------------------------------------------------------------------------


@flow(name="rag-eval-golden-flow", log_prints=True)
def eval_golden_flow(
    dataset_name: str | None = None,
    experiment_name: str | None = None,
    evaluator_names: list[str] | None = None,
) -> dict:
    """Deprecated wrapper for :func:`run_ground_truth_evaluation`.

    The registry-backed CE flow is canonical. This wrapper remains only to
    preserve existing imports and Prefect history.
    """
    if dataset_name or evaluator_names:
        _get_logger().warning(
            "eval_golden_flow is deprecated; dataset_name/evaluator_names are ignored. "
            "Use run_ground_truth_evaluation instead."
        )

    return run_ground_truth_evaluation.fn(experiment_name=experiment_name)


@flow(name="rag-eval-trace-flow", log_prints=True)
def eval_trace_flow(
    time_window_minutes: int | None = None,
    evaluator_names: list[str] | None = None,
) -> dict:
    """Deprecated wrapper for :func:`run_nightly_drift_detection`."""
    if evaluator_names:
        _get_logger().warning(
            "eval_trace_flow is deprecated; evaluator_names is ignored. "
            "Use run_nightly_drift_detection instead."
        )

    return run_nightly_drift_detection.fn(lookback_minutes=time_window_minutes)


# ---------------------------------------------------------------------------
# C) Synthetic QA generation
# ---------------------------------------------------------------------------


@flow(name="rag-synthetic-qa-flow", log_prints=True)
def synthetic_qa_flow(
    corpus_source: str = "document_store",
) -> dict:
    """Generate synthetic QA pairs and upload as a Phoenix Dataset."""
    eval_config = get_eval_config()
    log = _get_logger()

    _assert_eval_runtime_ready()

    import asyncio

    from rag_core.core.config import load_config
    from rag_core.core.readiness import ApplicationReadinessChecker
    from rag_core.core.telemetry import init_observability

    from evaluation.generation.synthetic import generate_synthetic_dataset

    # ── Guard: Ensure ingestion has run ──────────────────────────────
    try:
        ApplicationReadinessChecker().ensure_ready()
    except Exception:
        log.error(
            "\n\n" + "=" * 70 + "\n"
            "🚨 SYNTHETIC QA PRE-FLIGHT CHECK FAILED\n\n"
            "Your knowledge base is empty or uninitialized. You cannot generate\n"
            "synthetic QA pairs because there are no documents to read from.\n\n"
            "Please run this command first:\n"
            "  make ingest\n\n"
            "Once your documents are loaded, you can run generation again.\n"
            + "=" * 70 + "\n"
        )
        import sys
        sys.tracebacklimit = 0
        raise RuntimeError("Generation aborted: Knowledge base is empty. Please run 'make ingest' first.") from None

    init_observability(eval_config.phoenix_project)
    config = load_config()

    log.info("Generating synthetic QA dataset from '%s' …", corpus_source)
    dataset_name = asyncio.run(generate_synthetic_dataset(config, eval_config))

    log.info("Synthetic dataset generated: %s", dataset_name)
    return {"dataset_name": dataset_name}
