"""Continuous Evaluation Prefect flows.

These flows are intentionally non-blocking with respect to evaluation
quality: low scores and per-example evaluator failures are logged to
Phoenix and returned in the flow result, but they do not crash the flow.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from functools import partial
from typing import Any

from prefect import flow, get_run_logger
from prefect.exceptions import MissingContextError

from evaluation.config import get_eval_config
from evaluation.eval_registry import (
    DRIFT_EVALUATORS,
    GROUND_TRUTH_EVALUATORS,
    RagasMetricSpec,
)
from evaluation.evaluators.custom_judges import SourceReferenceFormatJudge
from rag_core.core.telemetry import (
    FAILED_VERIFICATION_SPAN_NAME,
    GENERATION_METRICS_SPAN_NAME,
)

logger = logging.getLogger(__name__)

_FAILED_VERIFICATION_FILTER = (
    f"name == '{FAILED_VERIFICATION_SPAN_NAME}'"
)
_GENERATION_METRICS_FILTER = (
    f"name == '{GENERATION_METRICS_SPAN_NAME}'"
)


def _get_logger() -> logging.Logger:
    """Return the Prefect run logger when available, else stdlib."""
    try:
        return get_run_logger()
    except MissingContextError:
        return logging.getLogger(__name__)


def _assert_eval_runtime_ready() -> None:
    """Validate optional eval/runtime dependencies at flow execution time."""
    from evaluation.runtime import assert_eval_runtime_ready

    assert_eval_runtime_ready()


def _check_eval_knowledge_base_ready(flow_label: str) -> object | None:
    """Return a failed Prefect state when eval is requested before ingestion."""
    from prefect.states import Failed
    from rag_core.core.readiness import ApplicationReadinessChecker, SetupRequiredError

    log = _get_logger()
    try:
        ApplicationReadinessChecker().ensure_ready()
        return None
    except SetupRequiredError as exc:
        log.error(
            "\n\n"
            + "=" * 70
            + "\n"
            + f"{flow_label} PRE-FLIGHT CHECK FAILED\n\n"
            + "The knowledge base is empty or not ready, so evaluation cannot run.\n"
            + "You need to ingest documents first.\n\n"
            + "Run this command:\n"
            + "  make ingest\n\n"
            + "Or use the Prefect dashboard:\n"
            + "  http://localhost:4200\n"
            + "  Deployment: ingest-documents\n\n"
            + "Reason:\n"
            + f"  {exc.message}\n"
            + "=" * 70
            + "\n"
        )
        return Failed(
            message=(
                "Evaluation aborted: knowledge base is empty or not ready. "
                "Run 'make ingest' first."
            )
        )


def _metric_names(evaluators: Iterable[object]) -> list[str]:
    """Extract Ragas metric names from registry entries."""
    return [item.name for item in evaluators if isinstance(item, RagasMetricSpec)]


def _custom_judges(evaluators: Iterable[object]) -> list[SourceReferenceFormatJudge]:
    """Extract deterministic custom judges from registry entries."""
    return [
        item
        for item in evaluators
        if isinstance(item, SourceReferenceFormatJudge)
    ]


def _build_rag_service() -> object:
    """Construct the production RAG service once per flow run."""
    from app.services.rag_service import RagService
    from rag_core.core.readiness import ApplicationReadinessChecker
    from rag_core.genetation.graph import RagGraphFactory
    from rag_core.prompts.loader import AgentPrompt

    graph = RagGraphFactory.build_default()
    agent_prompt = AgentPrompt.from_package("primary_rag_agent")
    return RagService(
        graph=graph,
        system_instructions=agent_prompt.get_system_instructions(),
        model_id="eval-runner",
        readiness_checker=ApplicationReadinessChecker(),
    )


def _build_evaluators(evaluator_registry: list[object]) -> list[object]:
    """Resolve registry entries into callable evaluator instances."""
    from evaluation.evaluators.ragas_metrics import build_ragas_evaluators
    from rag_core.core.config import load_config

    eval_config = get_eval_config()
    metric_names = _metric_names(evaluator_registry)
    config = load_config()

    evaluators: list[object] = []
    if metric_names:
        metric_config = eval_config.model_copy(update={"ragas_metrics": metric_names})
        evaluators.extend(build_ragas_evaluators(config, metric_config))
    evaluators.extend(_custom_judges(evaluator_registry))
    return evaluators


def _extract_score(value: object) -> float | None:
    """Extract a scalar score from common evaluator result shapes."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        score = value.get("score")
        if isinstance(score, (int, float)):
            return float(score)
    return None


def _safe_evaluate_row(
    *,
    evaluator: object,
    question: str,
    contexts: list[str],
    answer: str,
    ground_truth: str = "",
) -> dict[str, Any]:
    """Run one evaluator against one row without raising."""
    name = getattr(evaluator, "name", getattr(evaluator, "__name__", "evaluator"))
    try:
        if isinstance(evaluator, SourceReferenceFormatJudge):
            result = evaluator.evaluate(answer=answer)
        else:
            result = evaluator(
                input={"input": question},
                output={"response": answer, "retrieved_contexts": contexts},
                expected={"expected": ground_truth},
                metadata={},
            )

        score = _extract_score(result)
        if score is None:
            score = 0.0
        label = result.get("label") if isinstance(result, dict) else None
        return {"name": name, "score": score, "label": label, "error": None}
    except Exception as exc:
        _get_logger().exception("Evaluator '%s' failed", name)
        return {"name": name, "score": 0.0, "label": "error", "error": str(exc)}


def _load_ground_truth_rows(path: str, max_examples: int | None) -> list[dict[str, str]]:
    """Load and validate the static ground-truth CSV."""
    from evaluation.datasets.loaders import load_csv_rows
    from pathlib import Path

    # Resolve short names (e.g. "adversarial") to full paths.
    # This is the same logic as EvalConfig._resolve_dataset_path but acts
    # as a safety net when the caller bypasses the config validator.
    p = Path(path)
    if p.parent == Path("."):
        if not p.suffix:
            p = p.with_suffix(".csv")
        p = Path("evaluation/datasets") / p
    elif not p.suffix:
        p = p.with_suffix(".csv")
    path = str(p)

    rows = load_csv_rows(path)
    missing = [
        index
        for index, row in enumerate(rows, start=1)
        if "question" not in row or "ground_truth" not in row
    ]
    if missing:
        raise ValueError(
            "Ground-truth CSV rows must include 'question' and 'ground_truth'. "
            f"Invalid rows: {missing[:10]}"
        )
    if max_examples:
        rows = rows[:max_examples]
    return rows


def _normalise_contexts(contexts: object) -> list[str]:
    """Convert retrieved context payloads into a list of strings."""
    if contexts is None:
        return []
    if isinstance(contexts, str):
        return [contexts]
    if isinstance(contexts, list):
        normalised: list[str] = []
        for item in contexts:
            if isinstance(item, str):
                normalised.append(item)
            elif isinstance(item, dict):
                normalised.append(str(item.get("content") or item.get("text") or item))
            else:
                normalised.append(str(item))
        return normalised
    return [str(contexts)]


def _decode_jsonish(value: object) -> object:
    """Parse JSON strings when possible; otherwise return the original value."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _read_first(mapping: dict[str, Any], keys: Iterable[str], default: Any = "") -> Any:
    """Read the first present key from a dict."""
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return default


def _row_to_dict(row: object) -> dict[str, Any]:
    """Convert pandas rows, dicts, or SDK models to a plain dict."""
    if isinstance(row, dict):
        return row
    if hasattr(row, "to_dict"):
        return row.to_dict()
    return dict(getattr(row, "__dict__", {}))


def _extract_trace_example(row: object) -> dict[str, Any] | None:
    """Extract ``question``, ``contexts``, and ``answer`` from a Phoenix span."""
    row_dict = _row_to_dict(row)
    attributes = row_dict.get("attributes")
    if not isinstance(attributes, dict):
        attributes = {}

    merged = {**row_dict, **attributes}
    input_value = _decode_jsonish(
        _read_first(
            merged,
            ["question", "attributes.question", "input", "input.value", "llm.input_messages"],
            default="",
        )
    )
    output_value = _decode_jsonish(
        _read_first(
            merged,
            ["answer", "attributes.answer", "response", "output.value", "llm.output_messages"],
            default="",
        )
    )

    question = input_value
    if isinstance(input_value, dict):
        question = _read_first(input_value, ["question", "input", "content"], "")
    if isinstance(input_value, list) and input_value:
        question = str(input_value[-1])

    answer = output_value
    if isinstance(output_value, dict):
        answer = _read_first(output_value, ["answer", "response", "content"], "")
    if isinstance(output_value, list) and output_value:
        answer = str(output_value[-1])

    contexts = _normalise_contexts(
        _read_first(
            merged,
            [
                "contexts",
                "attributes.contexts",
                "retrieved_contexts",
                "output.retrieved_contexts",
                "retrieval.documents",
            ],
            default=[],
        )
    )

    question = str(question or "")
    answer = str(answer or "")
    if not question and not answer:
        return None

    return {
        "span_id": str(
            _read_first(merged, ["context.span_id", "span_id", "id"], default="")
        ),
        "trace_id": str(_read_first(merged, ["context.trace_id", "trace_id"], default="")),
        "question": question,
        "contexts": contexts,
        "answer": answer,
    }


def _query_generation_metrics_spans(
    *,
    client: object,
    project_name: str,
    start_time: datetime,
    end_time: datetime,
    limit: int,
) -> list[dict[str, Any]]:
    """Fetch ``agent.generation.metrics`` spans from Phoenix.

    Each returned example includes an ``intervention_count`` key extracted
    from the ``custom.intervention_count`` span attribute so the caller can
    stratify into high-risk vs baseline cohorts.
    """
    from phoenix.client.types.spans import SpanQuery

    query = SpanQuery().where(_GENERATION_METRICS_FILTER)
    spans_df = client.spans.get_spans_dataframe(
        query=query,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
        project_name=project_name,
    )

    examples: list[dict[str, Any]] = []
    if spans_df.empty:
        return examples

    for _, row in spans_df.iterrows():
        example = _extract_trace_example(row)
        if not example:
            continue
        row_dict = _row_to_dict(row)

        # Phoenix stores custom span attributes as a nested dict under
        # the key ``attributes.custom``, e.g.:
        #   {"attributes.custom": {"intervention_count": 2}}
        custom_block = row_dict.get("attributes.custom")
        if isinstance(custom_block, dict):
            raw = custom_block.get("intervention_count", 0)
        else:
            raw = 0

        try:
            example["intervention_count"] = int(raw)
        except (TypeError, ValueError):
            example["intervention_count"] = 0
        examples.append(example)

    return examples


def _annotate_spans(
    *,
    client: object,
    examples: list[dict[str, Any]],
    results: list[dict[str, Any]],
    cohort: str | None = None,
) -> int:
    """Write CE scores back to Phoenix spans when span IDs are available.

    When *cohort* is provided, an additional ``ce.cohort`` annotation is
    written so dashboards can segment metrics by sampling bucket.
    """
    inserted = 0
    for example, row_result in zip(examples, results, strict=False):
        span_id = example.get("span_id")
        if not span_id:
            continue
        for score in row_result["scores"]:
            client.spans.add_span_annotation(
                span_id=span_id,
                annotation_name=f"ce.{score['name']}",
                annotator_kind="CODE",
                label=score.get("label"),
                score=score.get("score"),
                explanation=score.get("error"),
                metadata={"flow": "run_nightly_drift_detection"},
                sync=False,
            )
            inserted += 1
        if cohort:
            client.spans.add_span_annotation(
                span_id=span_id,
                annotation_name="ce.cohort",
                annotator_kind="CODE",
                label=cohort,
                score=None,
                metadata={"flow": "run_nightly_drift_detection"},
                sync=False,
            )
            inserted += 1
        # Stamp intervention_count as a numeric annotation so it is
        # chartable in the Phoenix Metrics tab.
        ic = example.get("intervention_count")
        if ic is not None:
            client.spans.add_span_annotation(
                span_id=span_id,
                annotation_name="ce.intervention_count",
                annotator_kind="CODE",
                label=None,
                score=float(ic),
                metadata={"flow": "run_nightly_drift_detection"},
                sync=False,
            )
            inserted += 1
    return inserted


@flow(name="run-ground-truth-evaluation", log_prints=True)
def run_ground_truth_evaluation(
    dataset_csv_path: str | None = None,
    experiment_name: str | None = None,
    max_examples: int | None = None,
) -> dict[str, Any]:
    """Evaluate static CSV examples with ground-truth and reference-free metrics."""
    log = _get_logger()
    eval_config = get_eval_config()

    _assert_eval_runtime_ready()
    failed_state = _check_eval_knowledge_base_ready("EVAL")
    if failed_state is not None:
        return failed_state

    from orchestration.evaluation.tasks import (
        emit_artifact_task,
        resolve_dataset_task,
        run_experiment_task,
    )
    from evaluation.phoenix_bootstrap import get_phoenix_client
    from evaluation.pipeline.task import run_rag_for_example
    from rag_core.core.telemetry import init_observability

    csv_path = dataset_csv_path or eval_config.dataset_csv_path
    row_limit = max_examples or eval_config.max_examples
    rows = _load_ground_truth_rows(csv_path, row_limit)
    log.info("Loaded %d ground-truth examples from %s", len(rows), csv_path)

    init_observability(eval_config.phoenix_project)
    rag_service = _build_rag_service()
    task_fn = partial(run_rag_for_example, rag_service=rag_service)
    evaluators = _build_evaluators(GROUND_TRUTH_EVALUATORS)

    flow_eval_config = eval_config.model_copy(
        update={
            "dataset_csv_path": csv_path,
            "max_examples": row_limit,
            "thresholds": {},
        }
    )
    dataset_handle = resolve_dataset_task(flow_eval_config)
    client = get_phoenix_client()
    dataset = client.datasets.get_dataset(dataset=dataset_handle)

    exp_name = experiment_name or f"{eval_config.dataset_name}-evaluation-{int(time.time())}"
    experiment = run_experiment_task(
        dataset=dataset,
        task_fn=task_fn,
        evaluators=evaluators,
        experiment_name=exp_name,
        timeout=eval_config.timeout,
        max_retries=eval_config.max_retries,
        project_name=eval_config.phoenix_project,
    )

    try:
        emit_artifact_task(experiment, None, flow_eval_config, aggregates={})
    except Exception as exc:
        log.warning("Could not emit Prefect artifact for CE run: %s", exc)

    log.info("Ground-truth evaluation completed (experiment=%s)", exp_name)
    return {
        "status": "completed",
        "experiment_name": exp_name,
        "row_count": len(rows),
        "evaluators": [getattr(evaluator, "name", getattr(evaluator, "__name__", "")) for evaluator in evaluators],
    }



@flow(name="run-nightly-drift-detection", log_prints=True)
def run_nightly_drift_detection(
    lookback_minutes: int | None = None,
) -> dict[str, Any]:
    """Evaluate traces separating agent efficiency from system health.

    Queries ``agent.generation.metrics`` spans (stamped by the
    ``post_processing`` node) and stratifies into two cohorts:

    * **High-Risk** — spans where the agent retried
      (``custom.intervention_count > 0``)
    * **Baseline** — clean first-try spans
      (``custom.intervention_count == 0``)

    Both cohorts are evaluated with ``ce.faithfulness`` and
    ``ce.answer_relevancy``, and each span receives a ``ce.cohort``
    annotation so Phoenix dashboards can chart them separately.

    Agent efficiency is the sum of ``intervention_count`` across all
    spans in the window — no LLM judge required.
    """
    import random

    log = _get_logger()
    eval_config = get_eval_config()

    _assert_eval_runtime_ready()
    failed_state = _check_eval_knowledge_base_ready("EVAL DRIFT")
    if failed_state is not None:
        return failed_state

    from evaluation.phoenix_bootstrap import get_phoenix_client

    client = get_phoenix_client()
    window = lookback_minutes or eval_config.trace_lookback_minutes

    # Step A: Strict Time Bounding
    end_time = datetime.now(UTC)
    start_time = end_time - timedelta(minutes=window)

    project_name = eval_config.phoenix_project
    annotations = 0

    # Step B: Query all generation metrics spans in window
    log.info("Step B: Querying generation metrics spans...")
    fetch_limit = eval_config.retry_cohort_max_examples + eval_config.baseline_max_examples
    all_spans = _query_generation_metrics_spans(
        client=client,
        project_name=project_name,
        start_time=start_time,
        end_time=end_time,
        limit=max(fetch_limit * 5, 1000),
    )

    # Step C: Stratified sampling
    high_risk_pool = [s for s in all_spans if s["intervention_count"] > 0]
    baseline_pool = [s for s in all_spans if s["intervention_count"] == 0]

    high_risk = high_risk_pool[: eval_config.retry_cohort_max_examples]
    baseline = random.sample(
        baseline_pool,
        min(len(baseline_pool), eval_config.baseline_max_examples),
    ) if baseline_pool else []

    log.info(
        "Stratified sampling: %d high-risk (of %d), %d baseline (of %d)",
        len(high_risk),
        len(high_risk_pool),
        len(baseline),
        len(baseline_pool),
    )

    # Step D: Agent efficiency metric (no LLM judge)
    total_interventions = sum(s["intervention_count"] for s in all_spans)
    log.info(
        "Agent efficiency: %d total interventions across %d spans",
        total_interventions,
        len(all_spans),
    )

    # Step E: Run LLM judges on each cohort with ce.cohort tagging
    evaluators = _build_evaluators(DRIFT_EVALUATORS)

    def _evaluate_cohort(
        cohort_examples: list[dict[str, Any]],
        cohort_label: str,
    ) -> list[dict[str, Any]]:
        cohort_results: list[dict[str, Any]] = []
        for example in cohort_examples:
            scores = [
                _safe_evaluate_row(
                    evaluator=evaluator,
                    question=example["question"],
                    contexts=example["contexts"],
                    answer=example["answer"],
                )
                for evaluator in evaluators
            ]
            cohort_results.append(
                {
                    "span_id": example.get("span_id"),
                    "trace_id": example.get("trace_id"),
                    "cohort": cohort_label,
                    "scores": scores,
                }
            )
        if cohort_examples:
            nonlocal annotations
            annotations += _annotate_spans(
                client=client,
                examples=cohort_examples,
                results=cohort_results,
                cohort=cohort_label,
            )
        return cohort_results

    high_risk_results = _evaluate_cohort(high_risk, "high_risk")
    baseline_results = _evaluate_cohort(baseline, "baseline")
    all_results = high_risk_results + baseline_results

    # ── Prefect markdown artifact ────────────────────────────────────
    def _avg_score(results: list[dict], metric: str) -> str:
        vals = [
            s["score"]
            for r in results
            for s in r["scores"]
            if s.get("name") == metric and s.get("score") is not None
        ]
        return f"{sum(vals) / len(vals):.2f}" if vals else "n/a"

    artifact_md = (
        f"## Drift Detection Summary\n\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| Lookback window | {window} min |\n"
        f"| Total spans queried | {len(all_spans)} |\n"
        f"| Total interventions | **{total_interventions}** |\n"
        f"| High-risk evaluated | {len(high_risk)} (of {len(high_risk_pool)}) |\n"
        f"| Baseline evaluated | {len(baseline)} (of {len(baseline_pool)}) |\n"
        f"| Annotations written | {annotations} |\n\n"
        f"### Average Scores by Cohort\n\n"
        f"| Cohort | Faithfulness | Answer Relevancy |\n"
        f"|--------|-------------|-----------------|\n"
        f"| **High-Risk** | {_avg_score(high_risk_results, 'faithfulness')} "
        f"| {_avg_score(high_risk_results, 'answer_relevancy')} |\n"
        f"| **Baseline** | {_avg_score(baseline_results, 'faithfulness')} "
        f"| {_avg_score(baseline_results, 'answer_relevancy')} |\n"
    )
    try:
        from prefect.artifacts import create_markdown_artifact

        create_markdown_artifact(
            key="drift-detection-summary",
            markdown=artifact_md,
            description="Nightly drift detection results with cohort breakdown",
        )
    except Exception as exc:
        log.debug("Failed to create Prefect artifact: %s", exc)

    log.info(
        "Nightly drift detection completed "
        "(total_interventions=%d, high_risk_evaluated=%d, "
        "baseline_evaluated=%d, total_annotations=%d)",
        total_interventions,
        len(high_risk),
        len(baseline),
        annotations,
    )
    return {
        "status": "completed",
        "lookback_minutes": window,
        "total_spans": len(all_spans),
        "total_interventions": total_interventions,
        "high_risk_count": len(high_risk),
        "baseline_count": len(baseline),
        "annotation_count": annotations,
        "results": all_results,
    }
