"""PDF → numbered Markdown conversion via Docling.

.. deprecated::
    This module has moved to :mod:`rag_core.ingestion.processing.converter`.
    This shim will be removed in a future release.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "rag_core.converter is deprecated. "
    "Import from rag_core.ingestion.processing.converter instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the canonical location
from rag_core.ingestion.processing.converter import (  # noqa: F401, E402
    PAGE_TMPL,
    PAGE_PATTERN,
    convert_pdf_to_markdown,
    get_clean_name,
)
