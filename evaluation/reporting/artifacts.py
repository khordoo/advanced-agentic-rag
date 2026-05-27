"""Prefect artifact emitters for evaluation experiment results.

Top-level imports: stdlib + logging only.  ``prefect.artifacts`` and
``pandas`` are imported lazily.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def emit_experiment_artifact(
    experiment,
    pass_fail,
    eval_config,
    aggregates: dict[str, float] | None = None,
) -> None:
    """Emit a Prefect markdown artifact summarising the experiment.

    Sections:
      1. Phoenix Experiments UI link
      2. Aggregate metric table
      3. Threshold pass/fail column
      4. First 5 failed examples (if any)
    """
    from prefect.artifacts import create_markdown_artifact  # lazy

    from evaluation.phoenix_bootstrap import phoenix_experiment_url

    # ── Build markdown ───────────────────────────────────────────────
    exp_id = getattr(experiment, "id", "unknown")
    exp_url = phoenix_experiment_url(exp_id)

    lines: list[str] = [
        f"## 🧪 Evaluation Experiment: `{getattr(experiment, 'name', exp_id)}`",
        "",
        f"**Phoenix UI:** [{exp_url}]({exp_url})",
        "",
    ]

    # Aggregate metrics table
    if aggregates:
        lines.append("### Aggregate Metrics")
        lines.append("")
        lines.append("| Metric | Score | Threshold | Status |")
        lines.append("|--------|-------|-----------|--------|")

        thresholds = eval_config.thresholds or {}
        breach_metrics = {b[0] for b in pass_fail.breaches} if pass_fail else set()

        for metric, score in sorted(aggregates.items()):
            threshold = thresholds.get(metric, "—")
            if isinstance(threshold, float):
                threshold_str = f"{threshold:.4f}"
            else:
                threshold_str = str(threshold)

            if metric in breach_metrics:
                status = "❌ FAIL"
            elif metric in thresholds:
                status = "✅ PASS"
            else:
                status = "ℹ️ —"

            lines.append(f"| {metric} | {score:.4f} | {threshold_str} | {status} |")
        lines.append("")

    # Pass/fail summary
    if pass_fail:
        lines.append(f"### Threshold Gate: {'❌ FAILED' if pass_fail.failed else '✅ PASSED'}")
        lines.append("")
        if pass_fail.breaches:
            lines.append("**Breaches:**")
            for metric, actual, required in pass_fail.breaches:
                lines.append(f"- `{metric}`: {actual:.4f} < {required:.4f}")
            lines.append("")

    # Failed examples (first 5)
    try:
        if hasattr(experiment, "get_results"):
            results_df = experiment.get_results()
            if "error" in results_df.columns:
                failed = results_df[results_df["error"].notna()].head(5)
                if not failed.empty:
                    lines.append("### ⚠️ Failed Examples (first 5)")
                    lines.append("")
                    for _, row in failed.iterrows():
                        q = row.get("input", "?")
                        err = row.get("error", "unknown")
                        lines.append(f"- **Q:** {q[:100]}…  → `{err}`")
                    lines.append("")
    except Exception as exc:
        logger.debug("Could not extract failed examples: %s", exc)

    markdown = "\n".join(lines)

    try:
        create_markdown_artifact(
            key="eval-experiment-report",
            markdown=markdown,
            description=f"Evaluation report for experiment {exp_id}",
        )
        logger.info("Prefect markdown artifact created for experiment %s", exp_id)
    except Exception as exc:
        logger.warning("Failed to create Prefect artifact: %s", exc)
        # Print to stdout as fallback
        print(markdown)


def emit_csv_artifact(experiment, eval_config) -> None:
    """Optionally emit a CSV table artifact (when ``csv_artifact_enabled``).

    Prefers ``prefect.artifacts.create_table_artifact`` over filesystem
    dumps — no files written.
    """
    if not eval_config.csv_artifact_enabled:
        return

    try:
        from prefect.artifacts import create_table_artifact  # lazy

        if hasattr(experiment, "get_results"):
            df = experiment.get_results()
        elif hasattr(experiment, "to_pandas"):
            df = experiment.to_pandas()
        else:
            logger.warning("Cannot extract tabular data from experiment object")
            return

        # Convert to list-of-dicts for Prefect table artifact
        records = df.head(100).to_dict(orient="records")
        create_table_artifact(
            key="eval-experiment-csv",
            table=records,
            description="Evaluation results (CSV format)",
        )
        logger.info("Prefect CSV table artifact created (%d rows)", len(records))
    except Exception as exc:
        logger.warning("Failed to create CSV artifact: %s", exc)
