"""Typed configuration for the RAG evaluation pipeline.

All fields are feature-flagged OFF by default so a fresh clone with
``cp .env.example .env && make ingest`` never imports heavy eval deps.

Top-level imports are limited to stdlib + Pydantic + pydantic_settings.
No phoenix, ragas, or rag_core imports at module level.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

logger = logging.getLogger(__name__)


class EvalConfig(BaseSettings):
    """Strongly typed configuration for evaluation runs.

    Every field maps to an ``EVAL_*`` environment variable.  The master
    kill-switch is :pyattr:`enabled` — when *False* no heavy imports are
    executed, no deployments are registered, and no eval flows run.
    """

    # ── Feature flags ────────────────────────────────────────────────
    enabled: bool = Field(
        default=False,
        description="Master kill switch for the entire evaluation subsystem.",
    )
    schedule_cron: str | None = Field(
        default=None,
        description="Cron expression for scheduled ground truth evaluation.",
    )
    trace_schedule_cron: str | None = Field(
        default=None,
        description="Cron expression for online trace-based evaluation.",
    )
    post_ingestion_hook: bool = Field(
        default=False,
        description="Run eval_golden_flow at the end of ingestion (non-blocking).",
    )

    # ── Dataset ──────────────────────────────────────────────────────
    dataset_name: str = Field(
        default="rag-ground-truth-v1",
        description="Phoenix Dataset name for the ground truth evaluation set.",
    )
    dataset_csv_path: str = Field(
        default="evaluation/datasets/ground_truth_dataset.csv",
        description="Path to the bundled ground truth CSV relative to project root.",
    )

    # ── Phoenix / Observability ──────────────────────────────────────
    phoenix_project: str = Field(
        default="advanced-rag-eval",
        description="Phoenix project name for eval spans (isolated from prod).",
    )
    phoenix_endpoint: str | None = Field(
        default=None,
        description=(
            "Override for Phoenix collector endpoint.  Falls back to "
            "PHOENIX_COLLECTOR_ENDPOINT → OTEL_EXPORTER_OTLP_TRACES_ENDPOINT."
        ),
    )

    # ── Judge model ──────────────────────────────────────────────────
    judge_provider: str = Field(
        default="aws",
        description="LLM provider for judge models (aws, ollama, lmstudio, kimi).",
    )
    judge_model: str | None = Field(
        default="openai.gpt-oss-120b-1:0",
        description="Override the default judge model for the chosen provider.",
    )

    # ── Execution tuning ─────────────────────────────────────────────
    concurrency: int = Field(
        default=4,
        ge=1,
        description="Max parallel examples during Phoenix experiment execution.",
    )
    trace_lookback_minutes: int = Field(
        default=60,
        ge=1,
        description="Time window (minutes) for trace-based evaluation.",
    )
    csv_artifact_enabled: bool = Field(
        default=False,
        description="Emit an optional CSV Prefect artifact alongside markdown.",
    )
    max_examples: int | None = Field(
        default=None,
        ge=1,
        description="Hard cap on ground truth examples to bound judge cost.",
    )

    # ── Evaluator selection ──────────────────────────────────────────
    ragas_metrics: Annotated[list[str], NoDecode] = Field(
        default=[
            "faithfulness",
            "answer_relevancy",
            "context_recall",
            "context_precision",
            "answer_correctness",
        ],
        description="RAGAS metric names to run during ground truth evaluation.",
    )
    phoenix_evaluators: Annotated[list[str], NoDecode] = Field(
        default=["hallucination", "qa_correctness", "relevance"],
        description="Phoenix reference-free evaluator names.",
    )

    # ── Thresholds (pass/fail gate) ──────────────────────────────────
    thresholds: Annotated[dict[str, float], NoDecode] = Field(
        default_factory=dict,
        description=(
            "Metric → minimum score.  Empty dict = warn-only (never fails).  "
            "JSON string in env, e.g. EVAL_THRESHOLDS='{\"faithfulness\": 0.9}'."
        ),
    )

    # ── Validators ───────────────────────────────────────────────────
    @field_validator("thresholds", mode="before")
    @classmethod
    def _parse_thresholds_json(cls, value: object) -> dict[str, float]:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return {}
            return json.loads(value)
        if isinstance(value, dict):
            return value
        return {}

    @field_validator("max_examples", mode="before")
    @classmethod
    def _parse_optional_int(cls, value: object) -> int | None:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("ragas_metrics", "phoenix_evaluators", mode="before")
    @classmethod
    def _parse_csv_list(cls, value: object) -> list[str]:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return []
            if value.startswith("["):
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [str(v).strip() for v in parsed if str(v).strip()]
                return []
            return [v.strip() for v in value.split(",") if v.strip()]
        if isinstance(value, list):
            return value
        return []

    model_config = SettingsConfigDict(
        env_prefix="EVAL_",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )


@lru_cache
def get_eval_config() -> EvalConfig:
    """Return a cached :class:`EvalConfig` instance."""
    return EvalConfig()
