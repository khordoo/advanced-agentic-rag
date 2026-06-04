"""Typed configuration for the RAG evaluation pipeline.

All fields are lightweight by default so a fresh clone with
``cp .env.example .env && make ingest`` never imports heavy eval deps.

Top-level imports are limited to stdlib + Pydantic + pydantic_settings.
No phoenix, ragas, or rag_core imports at module level.
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import Annotated

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

logger = logging.getLogger(__name__)

# Default directory for evaluation dataset CSVs (relative to project/container root).
_DEFAULT_DATASET_DIR = "evaluation/datasets"


class EvalConfig(BaseSettings):
    """Strongly typed configuration for evaluation runs.

    Every field maps to an ``EVAL_*`` environment variable.  Evaluation
    deployments are always registered; optional runtime dependencies are
    checked when an eval flow is actually executed.
    """

    # ── Feature flags ────────────────────────────────────────────────
    schedule_cron: str | None = Field(
        default=None,
        validation_alias="EVAL_STATIC_SCHEDULE_CRON",
        description="Cron expression for scheduled ground truth evaluation.",
    )
    trace_schedule_cron: str | None = Field(
        default=None,
        validation_alias="EVAL_DRIFT_SCHEDULE_CRON",
        description="Cron expression for online trace-based evaluation.",
    )
    auto_run_after_ingestion: bool = Field(
        default=False,
        validation_alias="EVAL_STATIC_POST_INGESTION_HOOK",
        description=(
            "Run run_ground_truth_evaluation at the end of ingestion (non-blocking)."
        ),
    )

    # ── Dataset ──────────────────────────────────────────────────────
    dataset_name: str = Field(
        default="ground-truth-v1",
        validation_alias="EVAL_STATIC_DATASET_NAME",
        description="Phoenix Dataset name for the ground truth evaluation set.",
    )
    dataset_csv_path: str = Field(
        default="evaluation/datasets/ground_truth_dataset.csv",
        validation_alias="EVAL_STATIC_DATASET_CSV_PATH",
        description=(
            "Path to the bundled ground truth CSV relative to project root. "
            "Accepts a short name like 'adversarial' which resolves to "
            "evaluation/datasets/adversarial.csv automatically."
        ),
    )

    @field_validator("dataset_csv_path")
    @classmethod
    def _resolve_dataset_path(cls, v: str) -> str:
        """Resolve short dataset names to full paths.

        Accepts:
        - Full absolute path: ``/app/evaluation/datasets/adversarial.csv``
        - Relative path:      ``evaluation/datasets/adversarial.csv``
        - Short name:         ``adversarial`` → ``evaluation/datasets/adversarial.csv``
        - With/without .csv:  ``adversarial.csv`` or ``adversarial``
        """
        from pathlib import Path

        p = Path(v)
        # If it looks like a bare name (no directory separators), resolve it
        if p.parent == Path("."):
            if not p.suffix:
                p = p.with_suffix(".csv")
            p = Path(_DEFAULT_DATASET_DIR) / p
        # If it's a relative path without .csv extension, add it
        elif not p.suffix:
            p = p.with_suffix(".csv")
        return str(p)

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
    timeout: int = Field(
        default=120,
        ge=1,
        description="Maximum time in seconds to wait for a single metric evaluation.",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum retries for a failed metric evaluation.",
    )
    max_workers: int = Field(
        default=4,
        ge=1,
        description="Maximum concurrent evaluation tasks.",
    )
    concurrency: int = Field(
        default=4,
        ge=1,
        description="Deprecated alias for max_workers.",
    )
    trace_lookback_minutes: int = Field(
        default=60,
        ge=1,
        validation_alias="EVAL_DRIFT_LOOKBACK_MINUTES",
        description="Time window (minutes) for trace-based evaluation.",
    )
    csv_artifact_enabled: bool = Field(
        default=False,
        validation_alias="EVAL_STATIC_CSV_ARTIFACT_ENABLED",
        description="Emit an optional CSV Prefect artifact alongside markdown.",
    )
    max_examples: int | None = Field(
        default=None,
        ge=1,
        validation_alias="EVAL_STATIC_MAX_EXAMPLES",
        description="Hard cap on ground truth examples to bound judge cost.",
    )
    retry_cohort_max_examples: int = Field(
        default=100,
        ge=1,
        validation_alias="EVAL_DRIFT_HIGH_RISK_SAMPLE_MAX",
        description="Max high-risk sample size (intervention_count > 0) to evaluate per drift run.",
    )
    baseline_max_examples: int = Field(
        default=20,
        ge=1,
        validation_alias="EVAL_DRIFT_BASELINE_SAMPLE_MAX",
        description="Max baseline sample size (intervention_count == 0) to randomly sample per drift run.",
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
        description="Legacy Phoenix reference-free evaluator names.",
    )

    # ── Answer correctness tuning ────────────────────────────────────
    factuality_weight: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description=(
            "Weight for factual correctness in RAGAS AnswerCorrectness "
            "(0.0–1.0). Semantic similarity weight is automatically "
            "1.0 − this value."
        ),
    )

    @property
    def semantic_weight(self) -> float:
        """Derived semantic similarity weight (always sums to 1.0)."""
        return 1.0 - self.factuality_weight

    @property
    def answer_correctness_weights(self) -> list[float]:
        """Weights list for RAGAS ``AnswerCorrectness(weights=...)``."""
        return [self.factuality_weight, self.semantic_weight]

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

    @model_validator(mode="after")
    def _sync_eval_worker_settings(self) -> "EvalConfig":
        fields_set = getattr(self, "__pydantic_fields_set__", set())

        if (
            "EVAL_CONCURRENCY" in os.environ and "EVAL_MAX_WORKERS" not in os.environ
        ) or (self.concurrency != 4 and self.max_workers == 4):
            self.max_workers = self.concurrency
        elif "max_workers" in fields_set:
            self.concurrency = self.max_workers
        elif "concurrency" in fields_set:
            self.max_workers = self.concurrency

        return self

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
