"""Prefect @flow definitions for the evaluation pipeline.

Top-level imports: stdlib + prefect + evaluation.config only.
All heavy deps (RagService, RagGraphFactory, evaluator factories, Phoenix client APIs)
are imported **lazily inside flow bodies** after runtime checks.

Pattern mirrors ``orchestration/ingestion/flows.py``.
"""

from __future__ import annotations

import logging
import time

from prefect import flow, get_run_logger
from prefect.exceptions import MissingContextError

from evaluation.config import get_eval_config

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
# A) Scheduled offline eval (golden dataset)
# ---------------------------------------------------------------------------


@flow(name="rag-eval-golden-flow", log_prints=True)
def eval_golden_flow(
    dataset_name: str | None = None,
    experiment_name: str | None = None,
    evaluator_names: list[str] | None = None,
) -> dict:
    """Run a full evaluation experiment against the golden dataset.

    Returns a summary dict on success.
    """
    eval_config = get_eval_config()
    log = _get_logger()

    from rag_core.core.readiness import ApplicationReadinessChecker

    _assert_eval_runtime_ready()

    # ── Guard: Ensure ingestion has run ──────────────────────────────
    try:
        ApplicationReadinessChecker().ensure_ready()
    except Exception:
        log.error(
            "\n\n" + "=" * 70 + "\n"
            "🚨 EVALUATION PRE-FLIGHT CHECK FAILED\n\n"
            "Your knowledge base is empty or uninitialized. You cannot evaluate\n"
            "the RAG pipeline because there are no documents to retrieve.\n\n"
            "Please run this command first:\n"
            "  make ingest\n\n"
            "Once your documents are loaded, you can run 'make eval' again.\n"
            + "=" * 70 + "\n"
        )
        import sys
        sys.tracebacklimit = 0
        raise RuntimeError("Evaluation aborted: Knowledge base is empty. Please run 'make ingest' first.") from None

    # ── Heavy imports (after guard) ──────────────────────────────────
    from functools import partial

    from rag_core.core.config import load_config
    from rag_core.core.telemetry import init_observability
    from rag_core.genetation.graph import RagGraphFactory
    from rag_core.prompts.loader import AgentPrompt

    from app.services.rag_service import RagService

    from evaluation.evaluators.ragas_metrics import build_ragas_evaluators
    from evaluation.evaluators.phoenix_judges import build_phoenix_evaluators
    from evaluation.orchestration.tasks import (
        apply_thresholds_task,
        emit_artifact_task,
        resolve_dataset_task,
        run_experiment_task,
    )
    from evaluation.phoenix_bootstrap import get_phoenix_client

    # ── Initialise observability for eval project ────────────────────
    init_observability(eval_config.phoenix_project)

    config = load_config()

    # ── Build RagService + evaluators ONCE ───────────────────────────
    log.info("Constructing RagService for evaluation …")
    graph = RagGraphFactory.build_default()
    agent_prompt = AgentPrompt.from_package("primary_rag_agent")
    rag_service = RagService(
        graph=graph,
        system_instructions=agent_prompt.get_system_instructions(),
        model_id="eval-runner",
    )

    from evaluation.pipeline.task import run_rag_for_example

    task_fn = partial(run_rag_for_example, rag_service=rag_service)

    log.info("Building evaluators …")
    evaluators = []
    evaluators.extend(build_ragas_evaluators(config, eval_config))
    evaluators.extend(build_phoenix_evaluators(config, eval_config))

    # ── Override dataset / experiment names if provided ───────────────
    ds_name = dataset_name or eval_config.dataset_name
    exp_name = experiment_name or f"golden-eval-{int(time.time())}"

    # Override eval_config dataset_name for resolve_dataset_task
    eval_config_override = eval_config.model_copy(
        update={"dataset_name": ds_name}
    )

    # ── Execute pipeline stages ──────────────────────────────────────
    dataset_handle = resolve_dataset_task(eval_config_override)

    # Resolve the actual Phoenix dataset object for run_experiment
    client = get_phoenix_client()
    dataset = client.datasets.get_dataset(dataset=dataset_handle)

    experiment = run_experiment_task(
        dataset=dataset,
        task_fn=task_fn,
        evaluators=evaluators,
        experiment_name=exp_name,
        concurrency=eval_config.concurrency,
        project_name=eval_config.phoenix_project,
    )

    # ── Compute aggregates ───────────────────────────────────────────
    aggregates = {}
    try:
        if hasattr(experiment, "get_results"):
            results_df = experiment.get_results()
            for col in results_df.select_dtypes(include=["number"]).columns:
                aggregates[col] = float(results_df[col].mean())
    except Exception as exc:
        log.warning("Could not compute aggregates: %s", exc)

    # ── Apply thresholds ─────────────────────────────────────────────
    pass_fail = apply_thresholds_task(experiment, eval_config, aggregates)

    # ── Emit artifacts ───────────────────────────────────────────────
    emit_artifact_task(experiment, pass_fail, eval_config, aggregates)

    # ── Fail flow if thresholds breached ─────────────────────────────
    if pass_fail.failed and eval_config.thresholds:
        breach_str = "; ".join(
            f"{m}={a:.4f} < {r:.4f}" for m, a, r in pass_fail.breaches
        )
        raise RuntimeError(f"Threshold breached: {breach_str}")

    log.info("Golden evaluation completed ✅ (experiment=%s)", exp_name)
    return {
        "experiment_name": exp_name,
        "aggregates": aggregates,
        "passed": not pass_fail.failed,
    }


# ---------------------------------------------------------------------------
# B) Online trace-based eval
# ---------------------------------------------------------------------------


@flow(name="rag-eval-trace-flow", log_prints=True)
def eval_trace_flow(
    time_window_minutes: int | None = None,
    evaluator_names: list[str] | None = None,
) -> dict:
    """Run reference-free evaluators on recent production traces.

    Pulls spans from the Phoenix production project within the time window,
    runs phoenix.evals judges, and annotates the live spans.
    """
    eval_config = get_eval_config()
    log = _get_logger()

    _assert_eval_runtime_ready()

    # ── Placeholder for trace-based eval ─────────────────────────────
    # Full implementation depends on Phoenix span query API.
    window = time_window_minutes or eval_config.trace_lookback_minutes
    log.info(
        "Trace-based evaluation requested (lookback=%d min) — "
        "implementation pending Phoenix span query API integration",
        window,
    )

    return {"status": "not_implemented", "lookback_minutes": window}


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
