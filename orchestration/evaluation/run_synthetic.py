"""CLI bridge for running synthetic QA generation without traceback spam."""

from __future__ import annotations

import sys

from orchestration.evaluation.flows import synthetic_qa_flow


def main() -> None:
    state = synthetic_qa_flow(return_state=True)
    if state.is_failed():
        sys.exit(1)


if __name__ == "__main__":
    main()
