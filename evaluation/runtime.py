"""Runtime dependency checks for optional evaluation extras."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Iterable

from evaluation.ragas_compat import apply_ragas_compat

logger = logging.getLogger("evaluation.runtime")


class EvaluationRuntimeError(RuntimeError):
    """Raised when a requested eval flow is missing optional deps."""


def _missing_modules(module_names: Iterable[str]) -> list[str]:
    missing = []
    for module_name in module_names:
        try:
            importlib.import_module(module_name)
        except ImportError as exc:
            missing.append(exc.name or module_name)
    return missing


def assert_eval_runtime_ready() -> None:
    """Fail fast when the current image cannot run requested eval features."""
    apply_ragas_compat()

    required_modules = [
        "ragas",
        "phoenix.client",
        "phoenix.evals",
        "fastembed",
    ]

    missing = sorted(set(_missing_modules(required_modules)))
    if not missing:
        return

    missing_csv = ", ".join(missing)
    message = (
        "Evaluation runtime check failed. Missing optional dependencies: "
        f"{missing_csv}. Evaluation flows require the [eval] extras. "
        "Run `make eval` to install them and trigger the pipeline."
    )
    logger.critical(message)
    raise EvaluationRuntimeError(message)
