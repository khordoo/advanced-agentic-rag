"""Factory for RAGAS metric → Phoenix-compatible evaluator callables.

All heavy imports (ragas, langchain_huggingface, rag_core.agents) are
deferred into ``build_ragas_evaluators()`` — never at module level.

Each returned callable has the Phoenix evaluator signature::

    evaluator(input, output, expected, metadata) → float | dict
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Canonical metric name → RAGAS class + optional kwargs
_METRIC_MAP: dict[str, tuple[str, dict[str, Any]]] = {
    "faithfulness": ("Faithfulness", {}),
    "answer_relevancy": ("AnswerRelevancy", {}),
    "context_recall": ("ContextRecall", {}),
    "context_precision": ("ContextPrecision", {}),
    "answer_correctness": ("AnswerCorrectness", {"weights": [0.8, 0.2]}),
}


def build_ragas_evaluators(
    config: dict,
    eval_config,
) -> list[Callable]:
    """Build a list of Phoenix-compatible evaluator callables for RAGAS metrics.

    Singletons (judge LLM, embeddings) are built **once** per call and
    captured in closures — not per-example.

    Parameters
    ----------
    config:
        Application config dict (from ``load_config()``).
    eval_config:
        ``EvalConfig`` instance.

    Returns
    -------
    list[Callable]
        One callable per metric in ``eval_config.ragas_metrics``.
    """
    # ── Lazy imports ─────────────────────────────────────────────────
    from ragas import EvaluationDataset, evaluate  # noqa: F811
    from ragas.metrics import (
        AnswerCorrectness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.run_config import RunConfig
    from langchain_huggingface import HuggingFaceEmbeddings

    from rag_core.agents import AgentFactory
    from evaluation.pipeline.adapter import to_ragas_sample

    # ── Metric class lookup ──────────────────────────────────────────
    _class_map = {
        "Faithfulness": Faithfulness,
        "AnswerRelevancy": AnswerRelevancy,
        "ContextRecall": ContextRecall,
        "ContextPrecision": ContextPrecision,
        "AnswerCorrectness": AnswerCorrectness,
    }

    # ── Build singletons ONCE ────────────────────────────────────────
    judge_llm = AgentFactory.create(
        eval_config.judge_provider,
        temperature=0,
        model_name=eval_config.judge_model,
        config=config,
    )
    ragas_llm = LangchainLLMWrapper(judge_llm)

    hf_emb = HuggingFaceEmbeddings(
        model_name=config.get("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
    )
    ragas_emb = LangchainEmbeddingsWrapper(hf_emb)

    max_workers = int(config.get("LLM_MAX_PARALLEL_REQUEST", 3))
    run_config = RunConfig(timeout=60, max_retries=3, max_workers=max_workers)

    evaluators: list[Callable] = []

    for metric_name in eval_config.ragas_metrics:
        if metric_name not in _METRIC_MAP:
            logger.warning("Unknown RAGAS metric '%s' — skipping", metric_name)
            continue

        class_name, kwargs = _METRIC_MAP[metric_name]
        metric_cls = _class_map[class_name]
        metric_instance = metric_cls(**kwargs) if kwargs else metric_cls()

        # Capture metric in closure
        def _make_evaluator(metric, name):
            def evaluator(
                input: dict,  # noqa: A002
                output: dict,
                expected: dict | None = None,
                metadata: dict | None = None,
            ) -> float:
                """Evaluate a single example with RAGAS metric *name*."""
                # Guard: Phoenix may pass None or an Exception if the RAG task fails
                if not isinstance(output, dict):
                    logger.error("RAGAS evaluator received invalid output type: %s", type(output))
                    return 0.0

                sample = to_ragas_sample(
                    {"input": input.get("input", ""), "expected": (expected or {}).get("expected", "")},
                    {"response": output.get("response", ""), "retrieved_contexts": output.get("retrieved_contexts", [])},
                )
                try:
                    ds = EvaluationDataset.from_list([sample])
                    result = evaluate(
                        dataset=ds,
                        metrics=[metric],
                        llm=ragas_llm,
                        embeddings=ragas_emb,
                        run_config=run_config,
                    )
                    # Extract scalar score
                    scores = result.to_pandas()
                    if name in scores.columns:
                        return float(scores[name].iloc[0])
                    # Fallback: first numeric column
                    for col in scores.columns:
                        try:
                            return float(scores[col].iloc[0])
                        except (ValueError, TypeError):
                            continue
                    return 0.0
                except Exception as exc:
                    logger.error("RAGAS metric '%s' failed: %s", name, exc)
                    return 0.0

            evaluator.__name__ = name
            evaluator.__qualname__ = f"ragas_{name}"
            return evaluator

        evaluators.append(_make_evaluator(metric_instance, metric_name))
        logger.info("Registered RAGAS evaluator: %s", metric_name)

    return evaluators
