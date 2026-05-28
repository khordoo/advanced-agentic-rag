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
    FAILED_VERIFICATION_ATTRIBUTE,
    FAILED_VERIFICATION_VALUE,
)

logger = logging.getLogger(__name__)

_FAILED_VERIFICATION_FILTER = (
    f"attributes['{FAILED_VERIFICATION_ATTRIBUTE}'] == "
    f"'{FAILED_VERIFICATION_VALUE}'"
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
    from rag_core.genetation.graph import RagGraphFactory
    from rag_core.prompts.loader import AgentPrompt

    graph = RagGraphFactory.build_default()
    agent_prompt = AgentPrompt.from_package("primary_rag_agent")
    return RagService(
        graph=graph,
        system_instructions=agent_prompt.get_system_instructions(),
        model_id="eval-runner",
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
            ["question", "input", "input.value", "llm.input_messages"],
            default="",
        )
    )
    output_value = _decode_jsonish(
        _read_first(
            merged,
            ["answer", "response", "output.value", "llm.output_messages"],
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


def _query_failed_verification_spans(
    *,
    client: object,
    project_name: str,
    lookback_minutes: int,
    limit: int,
) -> list[dict[str, Any]]:
    """Fetch Phoenix spans tagged as failed verifier attempts."""
    from phoenix.client.types.spans import SpanQuery

    end_time = datetime.now(UTC)
    start_time = end_time - timedelta(minutes=lookback_minutes)
    query = SpanQuery().where(_FAILED_VERIFICATION_FILTER)

    spans_df = client.spans.get_spans_dataframe(
        query=query,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
        project_name=project_name,
    )

    examples: list[dict[str, Any]] = []
    for _, row in spans_df.iterrows():
        example = _extract_trace_example(row)
        if example:
            examples.append(example)
    return examples


def _annotate_spans(
    *,
    client: object,
    examples: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> int:
    """Write CE scores back to Phoenix spans when span IDs are available."""
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

    from evaluation.orchestration.tasks import (
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

    exp_name = experiment_name or f"ground-truth-evaluation-{int(time.time())}"
    experiment = run_experiment_task(
        dataset=dataset,
        task_fn=task_fn,
        evaluators=evaluators,
        experiment_name=exp_name,
        concurrency=eval_config.concurrency,
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
    limit: int = 1000,
) -> dict[str, Any]:
    """Evaluate failed verifier traces with reference-free drift metrics."""
    log = _get_logger()
    eval_config = get_eval_config()

    _assert_eval_runtime_ready()

    from evaluation.phoenix_bootstrap import get_phoenix_client

    client = get_phoenix_client()
    window = lookback_minutes or eval_config.trace_lookback_minutes
    evaluators = _build_evaluators(DRIFT_EVALUATORS)

    try:
        examples = _query_failed_verification_spans(
            client=client,
            project_name=eval_config.phoenix_project,
            lookback_minutes=window,
            limit=limit,
        )
    except Exception as exc:
        log.warning("Could not query Phoenix failed-verification spans: %s", exc)
        return {
            "status": "query_failed",
            "lookback_minutes": window,
            "error": str(exc),
        }

    results: list[dict[str, Any]] = []
    for example in examples:
        scores = [
            _safe_evaluate_row(
                evaluator=evaluator,
                question=example["question"],
                contexts=example["contexts"],
                answer=example["answer"],
            )
            for evaluator in evaluators
        ]
        results.append(
            {
                "span_id": example.get("span_id"),
                "trace_id": example.get("trace_id"),
                "scores": scores,
            }
        )

    annotations = 0
    try:
        annotations = _annotate_spans(
            client=client,
            examples=examples,
            results=results,
        )
    except Exception as exc:
        log.warning("Could not annotate Phoenix spans with CE scores: %s", exc)

    log.info(
        "Nightly drift detection completed (examples=%d, annotations=%d)",
        len(examples),
        annotations,
    )
    return {
        "status": "completed",
        "lookback_minutes": window,
        "example_count": len(examples),
        "annotation_count": annotations,
        "results": results,
    }
