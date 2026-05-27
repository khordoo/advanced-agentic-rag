"""Typed configuration for Prefect orchestration runtime."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PrefectConfig(BaseSettings):
    """Strongly typed configuration for Prefect orchestration."""

    api_url: str = Field(
        default="http://localhost:4200/api",
        validation_alias="PREFECT_API_URL",
    )
    work_pool_name: str = Field(
        default="rag-work-pool",
        validation_alias="PREFECT_INGESTION_WORK_POOL",
    )
    work_pool_type: str = "process"
    deployment_name: str = Field(
        default="ingest-documents",
        validation_alias="PREFECT_DEPLOYMENT_NAME",
    )
    max_wait_seconds: int = Field(
        default=120,
        validation_alias="PREFECT_INIT_MAX_WAIT",
    )
    initial_backoff: float = 1.0
    max_backoff: float = 10.0

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )


@lru_cache
def get_prefect_config() -> PrefectConfig:
    return PrefectConfig()
