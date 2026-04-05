"""Centralised configuration loader for advanced-rag."""

from __future__ import annotations

import os
import logging
from pathlib import Path
from dotenv import dotenv_values

logger = logging.getLogger(__name__)

# Keys that MUST exist in the environment / .env
_REQUIRED_KEYS: list[str] = []

# Sensible defaults for optional keys
_DEFAULTS: dict[str, str] = {
    "CHROMA_COLLECTION_NAME": "docs",
    "CHROMA_PERSIST_DIRECTORY": "artifacts/chroma_db",
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
    # Qdrant (production track)
    "QDRANT_URL": "http://localhost:6333",
    "QDRANT_COLLECTION_NAME": "docs",
    # LangFuse (production track)
    "LANGFUSE_HOST": "http://localhost:3000",
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
