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
from evaluation.orchestration.flows import (
    eval_golden_flow,
    eval_trace_flow,
    synthetic_qa_flow,
)

logger = logging.getLogger(__name__)


def build_eval_deployment_specs(eval_config: EvalConfig) -> list[tuple]:
    """Return deployment specs analogous to ``_DEPLOYMENT_SPECS`` in initialize.py.

    Each tuple: ``(flow_object, deployment_name, tags, description, schedule_cron)``.
    """
    specs: list[tuple] = [
        (
            eval_golden_flow,
            "evaluate-ground-truth",
            ["evaluation", "ground-truth", "oss"],
            "Evaluate RAG answers against the ground truth dataset.",
            eval_config.schedule_cron,
        ),
        (
            eval_trace_flow,
            "evaluate-traces",
            ["evaluation", "trace", "oss"],
            "Evaluate recent production traces with reference-free judges.",
            eval_config.trace_schedule_cron,
        ),
        (
            synthetic_qa_flow,
            "generate-synthetic-qa",
            ["evaluation", "synthetic", "oss"],
            "Generate synthetic question-answer examples from indexed documents.",
            None,  # No default schedule for synthetic generation
        ),
    ]

    logger.info("Built %d eval deployment specs", len(specs))
    return specs
