"""Markdown → parent / child chunk splitting and storage.

.. deprecated::
    This module has moved to :mod:`rag_core.ingestion.processing.chunker`.
    This shim will be removed in a future release.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "rag_core.chunker is deprecated. "
    "Import from rag_core.ingestion.processing.chunker instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the canonical location
from rag_core.ingestion.processing.chunker import (  # noqa: F401, E402
    HEADERS_TO_SPLIT_ON,
    HEADER_KEYS,
    chunk_markdown,
    summerize_and_chunk_table,
)
