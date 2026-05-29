"""Lazy Phoenix client factory and URL helpers.

Top-level imports: stdlib only.  Phoenix SDK imports are deferred inside
functions so this module is safe to import without the ``[eval]`` extra.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def _resolve_phoenix_endpoint() -> str:
    """Resolve the Phoenix HTTP endpoint from available env / config.

    Priority:
      1. ``EvalConfig.phoenix_endpoint`` (via caller)
      2. ``PHOENIX_COLLECTOR_ENDPOINT`` env var
      3. ``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`` with ``/v1/traces`` stripped
      4. Fallback: ``http://localhost:6006``
    """
    from evaluation.config import get_eval_config

    eval_cfg = get_eval_config()
    if eval_cfg.phoenix_endpoint:
        return eval_cfg.phoenix_endpoint.rstrip("/")

    collector = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
    if collector:
        return collector.rstrip("/")

    otlp = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "")
    if otlp:
        # Strip the /v1/traces suffix to get the base Phoenix URL
        base = otlp.rstrip("/")
        if base.endswith("/v1/traces"):
            base = base[: -len("/v1/traces")]
        return base

    return "http://localhost:6006"


def get_phoenix_client():
    """Return a Phoenix SDK client pointed at the resolved endpoint.

    Raises
    ------
    ImportError
        If ``arize-phoenix-client`` is not installed.
    ConnectionError
        If Phoenix is unreachable at the resolved endpoint.
    """
    from phoenix.client import Client  # lazy

    endpoint = _resolve_phoenix_endpoint()
    logger.info("Connecting to Phoenix at %s", endpoint)
    return Client(base_url=endpoint)


def phoenix_experiment_url(experiment_id: str) -> str:
    """Build a clickable Phoenix Experiments UI URL."""
    base = _resolve_phoenix_endpoint()
    return f"{base}/experiments/{experiment_id}"
