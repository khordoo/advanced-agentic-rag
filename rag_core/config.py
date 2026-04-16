"""Centralised configuration loader for advanced-rag."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

import torch
from pydantic import computed_field
from typing import Literal
from dotenv import dotenv_values, load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# Keys that MUST exist in the environment / .env
_REQUIRED_KEYS: list[str] = []

# Sensible defaults for optional keys
_DEFAULTS: dict[str, str] = {
    "CHROMA_COLLECTION_NAME": "docs",
    "CHROMA_PERSIST_DIRECTORY": "artifacts/chroma_db",
    "DOCS_DIR": "docs",
    "PROCCESED_DOCS_DIR": "data/processed",
    "DOCUMENT_STORE_PERSIST_PATH": "artifacts/document_store/document_store.pkl",
    "SYNTHETIC_QA_PERSIST_PATH": "data/synthetic/eval_question_answers.json",
    "HF_EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
    "HF_RERANKER_MODEL": "BAAI/bge-reranker-v2-m3",
    "LLM_MAX_PARALLEL_REQUEST": "3",
    "PARENT_CHUNK_SIZE": "1500",
    "PARENT_CHUNK_OVERLAP": "150",
    "CHILD_CHUNK_SIZE": "300",
    "CHILD_CHUNK_OVERLAP": "50",
    "RERANK_TOP_N": "3",
    "TABLE_CONTEXT_WINDOW": "3",
    "SUMMARIZE_TABLES_ENABLED": "True",
    # Qdrant (production track)
    "QDRANT_URL": "http://localhost:6333",
    "QDRANT_COLLECTION_NAME": "docs",
    # LangFuse (production track)
    "LANGFUSE_HOST": "http://localhost:3000",
    "LOG_LEVEL": "INFO",
}


def load_config(env_path: str | Path | None = None) -> dict[str, str]:
    """Load configuration from a ``.env`` file and merge with OS env vars.

    Priority: OS env vars > .env file > defaults.

    Parameters
    ----------
    env_path:
        Path to the ``.env`` file.  When *None* the loader walks up from the
        current working directory looking for a ``.env`` file.
    """
    if env_path is None:
        # Walk up from cwd looking for .env
        cwd = Path.cwd()
        for parent in [cwd, *cwd.parents]:
            candidate = parent / ".env"
            if candidate.is_file():
                env_path = candidate
                break

    file_values = dotenv_values(env_path) if env_path else {}

    # Merge: defaults ← .env ← OS env
    config: dict[str, str] = {**_DEFAULTS, **file_values}
    for key in list(config.keys()) + list(_DEFAULTS.keys()):
        env_val = os.environ.get(key)
        if env_val is not None:
            config[key] = env_val

    # Handle legacy typos from old .env files
    _compat_aliases = {
        "PARENT_CHUNK_VOERLAP": "PARENT_CHUNK_OVERLAP",
        "CHILD_CHUNK_VOERLAP": "CHILD_CHUNK_OVERLAP",
    }
    for old_key, new_key in _compat_aliases.items():
        if old_key in config and new_key not in file_values:
            config[new_key] = config[old_key]

    # Validate required
    missing = [k for k in _REQUIRED_KEYS if k not in config]
    if missing:
        raise EnvironmentError(
            f"Missing required config keys: {', '.join(missing)}. "
            f"Add them to your .env file or set as environment variables."
        )

    logger.debug("Config loaded (%d keys)", len(config))
    return config


def setup_logging():
    """Independent logging setup that reads directly from the environment."""
    load_dotenv()
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


class SecuritySettings(BaseSettings):
    """Strictly for RBAC, scopes, and access control baselines."""

    default_scopes: list[str] = ["public"]

    # You can naturally expand this later without polluting your ingestion logic:
    admin_role: str = "admin"
    enable_strict_mode: bool = True

    model_config = SettingsConfigDict(
        env_prefix="SECURITY_", env_file=".env", extra="ignore"
    )


class RetrivalSettings(BaseSettings):
    """
    Configuration settings for the RAG retrieval pipeline.

    ⚠️ CRITICAL WARNING REGARDING THRESHOLDS:
    This system supports both Standard (Cosine) and Hybrid (RRF) search.
    They use completely incompatible scoring scales:
    - Cosine Similarity scores usually range from 0.0 to 1.0 (good matches are > 0.7).
    - RRF (Reciprocal Rank Fusion) ignores metric scores and calculates artificial
        scores based purely on rank: Score = 1 / (Rank + 60).
        Because of this math, the absolute #1 best RRF result will score ~0.016.

    Applying a standard cosine threshold (e.g., 0.5) to an RRF search will
    silently drop ALL documents and return empty results.
    """

    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Final number of parent documents to return to the LLM agent",
    )
    oversample_factor: int = Field(
        default=5,
        ge=1,
        le=5,
        description="Multiplier for initial child-chunk retrieval. Determines how many chunks to fetch before grouping by parent.",
    )
    hybrid_alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Weighting for hybrid search if using weighted fusion instead of RRF. "
            "0.0 = Sparse (BM25) only, 1.0 = Dense (Vectors) only. "
            "Most robust RAG systems perform best around 0.4 to 0.7."
        ),
    )
    # Hybrid Search Weights (0.0 = Sparse only, 1.0 = Dense only)
    # Most RAG systems perform best around 0.4 to 0.7
    score_threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum similarity score for standard DENSE searches (e.g., Cosine/Dot Product). "
            "Results scoring lower than this will be discarded. "
            "WARNING: Do NOT apply this to RRF hybrid queries."
        ),
    )
    rrf_score_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum score threshold for Qdrant RRF hybrid search. "
            "Qdrant calculates RRF as 1/(Rank+2). The max possible score is 1.0 (Rank #1 in both searches). "
            "A score of 0.10 means the document was roughly Rank #9 in one search and missing from the other. "
            "Set to ~0.05 - 0.08 to filter out low-confidence tail results."
        ),
    )
    fetch_parent: bool = True


class RerankerSettings(BaseSettings):
    """Configuration settings for the Reranker.
    We set top_k=10 since childs chunks will have duplicate parent chunks
    and we want to return at least 5 unique parent chunks to the LLM agent
    """

    provider: str = "huggingface"
    model_name: str = "BAAI/bge-reranker-v2-m3"
    top_k: int = Field(
        default=3,
        ge=1,
        le=50,
        description="Number of documents to return afterrerank",
    )
    device: Literal["cuda", "cpu", "mps", "auto"] = Field(
        default="auto",
        description="Device to use for reranking",
        examples=["cuda", "cpu", "mps", "auto"],
    )
    model_config = SettingsConfigDict(
        env_prefix="RERANK_", env_file=".env", extra="ignore"
    )

    @computed_field
    @property
    def resolve_device(self) -> Literal["cuda", "cpu", "mps"]:
        if self.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return self.device


class QdrantSettings(BaseSettings):
    url: str = "http://localhost:6333"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "prithivida/Splade_PP_en_v1"
    collection_namespace: str = "docs"
    chunk_collection_name: str = f"{collection_namespace}_chunks"
    parent_collection_name: str = f"{collection_namespace}_parents"
    model_config = SettingsConfigDict(
        env_prefix="QDRANT_", env_file=".env", extra="ignore"
    )


class PostgresSettings(BaseSettings):
    user: str = "admin"
    password: str = "password"
    host: str = "localhost"  # 'core-db' when running inside Docker
    port: int = 5432
    rag_db: str = "rag_registry"

    # Pydantic will ONLY look for variables starting with "POSTGRES_"
    model_config = SettingsConfigDict(
        env_prefix="POSTGRES_", env_file=".env", extra="ignore"
    )


# --- PIPELINE SETTINGS ---


class IngestionSettings(BaseSettings):
    """Strictly for tuning how the pipeline processes data."""

    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_retries: int = 3
    ALLOWED_EXTENSIONS: set = {".pdf", ".md", ".txt"}
    # You can add things like 'enable_ocr' or 'target_folder' here

    model_config = SettingsConfigDict(
        env_prefix="INGESTION_", env_file=".env", extra="ignore"
    )


@lru_cache
def get_qdrant_settings() -> QdrantSettings:
    logger.info("Reading .env file and caching Qdrant settings...")
    return QdrantSettings()


@lru_cache
def get_postgres_settings() -> PostgresSettings:
    logger.info("Reading .env file and caching Qdrant settings...")
    return PostgresSettings()


@lru_cache()
def get_ingestion_settings() -> IngestionSettings:
    logger.info("Reading .env file and caching Indexing pipeline settings...")
    return IngestionSettings()
