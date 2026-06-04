"""Declarative deployment specs for evaluation flows.

Consumed by ``orchestration/initialize.py``.  Eval deployments are
registered unconditionally; optional eval dependencies are checked when
the corresponding flow executes.

Top-level imports: evaluation.config + flow objects (both safe without
the ``[eval]`` extra installed).
"""

from __future__ import annotations

import logging

from evaluation.config import EvalConfig
from orchestration.evaluation.flows import synthetic_qa_flow
from orchestration.evaluation.prefect_flows import (
    run_ground_truth_evaluation,
    run_nightly_drift_detection,
)

logger = logging.getLogger(__name__)


def build_eval_deployment_specs(eval_config: EvalConfig) -> list[tuple]:
    """Return deployment specs analogous to ``_DEPLOYMENT_SPECS`` in initialize.py.

    Each tuple:
    ``(flow_object, deployment_name, tags, description, schedule_cron, flow_module_path)``.
    """
    specs: list[tuple] = [
        (
            run_ground_truth_evaluation,
            "evaluate-ground-truth",
            ["evaluation", "ground-truth", "oss"],
            "Evaluate RAG answers against the ground truth dataset.",
            eval_config.schedule_cron,
            "orchestration/evaluation/prefect_flows.py",
        ),
        (
            run_nightly_drift_detection,
            "evaluate-drift",
            ["evaluation", "drift", "oss"],
            "Evaluate recent production traces with reference-free judges.",
            eval_config.trace_schedule_cron,
            "orchestration/evaluation/prefect_flows.py",
        ),
        (
            synthetic_qa_flow,
            "generate-synthetic-qa",
            ["evaluation", "synthetic", "oss"],
            "Generate synthetic question-answer examples from indexed documents.",
            None,  # No default schedule for synthetic generation
            "orchestration/evaluation/flows.py",
        ),
    ]

    logger.info("Built %d eval deployment specs", len(specs))
    return specs
