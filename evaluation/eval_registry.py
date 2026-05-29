"""Central evaluator registry for Continuous Evaluation flows.

The registry stores lightweight evaluator specs so importing this module
does not require optional evaluation dependencies such as Ragas or Phoenix.
Runtime flow code resolves these specs after the eval dependency check passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from evaluation.evaluators.custom_judges import SourceReferenceFormatJudge


class EvaluatorProtocol(Protocol):
    """Minimal protocol shared by deterministic custom judges."""

    name: str


@dataclass(frozen=True)
class RagasMetricSpec:
    """Lazy reference to a Ragas metric by canonical project name."""

    name: str


# Reference-dependent metrics require a static reference answer
# (``ground_truth``) from the curated CSV evaluation dataset.
GROUND_TRUTH_EVALUATORS: list[RagasMetricSpec | EvaluatorProtocol] = [
    RagasMetricSpec("context_precision"),
    RagasMetricSpec("context_recall"),
    RagasMetricSpec("faithfulness"),
    RagasMetricSpec("answer_relevancy"),
    RagasMetricSpec("answer_correctness"),
    SourceReferenceFormatJudge(),
]


# Reference-free metrics run against live production traces where no
# ``ground_truth`` exists. They only require ``question``, ``contexts``,
# and ``answer``.
DRIFT_EVALUATORS: list[RagasMetricSpec | EvaluatorProtocol] = [
    RagasMetricSpec("faithfulness"),
    RagasMetricSpec("answer_relevancy"),
    SourceReferenceFormatJudge(),
]


__all__ = [
    "DRIFT_EVALUATORS",
    "GROUND_TRUTH_EVALUATORS",
    "RagasMetricSpec",
]
