"""Lightweight embedding provider types."""

from __future__ import annotations

from enum import Enum


class EmbeddingProvider(str, Enum):
    """Supported dense embedding provider adapters."""

    FASTEMBED = "fastembed"
    OPENAI = "openai"
