"""Pass/fail threshold policy for evaluation metrics.

No heavy imports — stdlib ``dataclasses`` + ``logging`` only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PassFailReport:
    """Result of applying threshold gates to aggregate metrics.

    Attributes
    ----------
    failed:
        ``True`` if any metric breached its configured threshold.
    breaches:
        List of ``(metric_name, actual_score, required_threshold)`` tuples
        for metrics that fell below their threshold.
    """

    failed: bool = False
    breaches: list[tuple[str, float, float]] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary for logging and Prefect artifacts."""
        if not self.failed:
            return "✅ All thresholds passed"
        lines = ["❌ Threshold breaches detected:"]
        for metric, actual, required in self.breaches:
            lines.append(f"  - {metric}: {actual:.4f} < {required:.4f}")
        return "\n".join(lines)


def evaluate_thresholds(aggregates: dict[str, float], eval_config) -> PassFailReport:
    """Apply ``eval_config.thresholds`` to aggregate metric scores.

    Parameters
    ----------
    aggregates:
        Metric name → aggregate score (mean) from the experiment.
    eval_config:
        ``EvalConfig`` instance.

    Returns
    -------
    PassFailReport
        ``failed=False`` when thresholds dict is empty (warn-only mode).
    """
    thresholds = eval_config.thresholds
    if not thresholds:
        logger.info("No thresholds configured — warn-only mode (never fails)")
        return PassFailReport(failed=False)

    breaches: list[tuple[str, float, float]] = []

    for metric_name, min_score in thresholds.items():
        actual = aggregates.get(metric_name)
        if actual is None:
            logger.warning(
                "Threshold metric '%s' not found in aggregates — skipping",
                metric_name,
            )
            continue
        if actual < min_score:
            logger.warning(
                "Threshold breach: %s=%.4f < %.4f",
                metric_name,
                actual,
                min_score,
            )
            breaches.append((metric_name, actual, min_score))
        else:
            logger.info(
                "Threshold passed: %s=%.4f >= %.4f",
                metric_name,
                actual,
                min_score,
            )

    report = PassFailReport(failed=len(breaches) > 0, breaches=breaches)
    logger.info(report.summary())
    return report
