"""CLI bridge for running drift evaluation without traceback spam."""

from __future__ import annotations

import sys

from orchestration.evaluation.prefect_flows import run_nightly_drift_detection


def main() -> None:
    state = run_nightly_drift_detection(return_state=True)
    if state.is_failed():
        sys.exit(1)


if __name__ == "__main__":
    main()
