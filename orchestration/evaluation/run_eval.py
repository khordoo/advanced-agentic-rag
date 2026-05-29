"""CLI bridge for running ground-truth evaluation without traceback spam."""

from __future__ import annotations

import sys

from orchestration.evaluation.prefect_flows import run_ground_truth_evaluation


def main() -> None:
    state = run_ground_truth_evaluation(return_state=True)
    if state.is_failed():
        sys.exit(1)


if __name__ == "__main__":
    main()
