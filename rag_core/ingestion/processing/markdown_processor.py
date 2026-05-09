"""Markdown-based asset processor.

Composes PDF/Markdown conversion, hierarchical chunking, and table
summarisation into a single :meth:`process_asset` call that fulfils the
:class:`BaseProcessor` contract.
"""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

from rag_core.agents import TableSummarizer
from rag_core.ingestion.processing.base import BaseProcessor
from rag_core.ingestion.processing.converter import convert_pdf_to_markdown
from rag_core.ingestion.processing.chunker import chunk_markdown, summerize_and_chunk_table
from rag_core.schemas import ClaimedSourceAsset, ParentChunk

logger = logging.getLogger(__name__)


def _resolve_local_path(source_uri: str) -> str:
    """Convert a source URI (possibly ``file://``) to an absolute OS path."""
    parsed = urlparse(source_uri)
    if parsed.scheme == "file":
        if parsed.netloc == ".":
            return str(Path("." + parsed.path).resolve())
        return url2pathname(parsed.path)
    # Plain filesystem path (no scheme)
    return str(Path(source_uri).resolve())


class MarkdownProcessor(BaseProcessor):
    """Production processor: convert → chunk → summarise tables.

    This processor handles ``.pdf`` and ``.md`` files.  It delegates to
    the same battle-tested functions previously called by the legacy
    ``IngestionPipeline``, but wraps them behind the ``BaseProcessor``
    contract so they can be injected into
    :class:`~rag_core.ingestion.services.worker.SourceIngestionWorkerService`.

    Parameters
    ----------
    table_summarizer:
        LLM-backed summariser for extracted tables.  When *None*, a
        default ``TableSummarizer`` is constructed (using the AWS provider).
    summarize_tables_enabled:
        Toggle LLM-based table summarisation.
    table_context_window:
        Number of surrounding text elements to capture around each table
        during PDF conversion.
    config:
        Optional config dict for chunk-size overrides (``PARENT_CHUNK_SIZE``,
        ``CHILD_CHUNK_SIZE``, etc.).
    """

    def __init__(
        self,
        table_summarizer: TableSummarizer | None = None,
        *,
        summarize_tables_enabled: bool = True,
        table_context_window: int = 3,
        config: dict | None = None,
    ) -> None:
        self.table_summarizer = table_summarizer or TableSummarizer()
        self.summarize_tables_enabled = summarize_tables_enabled
        self.table_context_window = table_context_window
        self.config = config or {}

    def process_asset(self, asset: ClaimedSourceAsset) -> list[ParentChunk]:
        """Convert a claimed source asset into parent/child chunks.

        Steps
        -----
        1. Resolve the ``source_uri`` to a local file path.
        2. Convert to Markdown (PDF) or read directly (.md).
        3. Extract and optionally summarise tables.
        4. Chunk the Markdown into hierarchical parent/child structures.
        5. Merge table chunks with text chunks.
        """
        absolute_path = _resolve_local_path(asset.source_uri)
        file_ext = Path(absolute_path).suffix.lower()

        logger.info(
            "MarkdownProcessor starting",
            extra={
                "asset_id": asset.id,
                "source_uri": asset.source_uri,
                "resolved_path": absolute_path,
                "file_ext": file_ext,
            },
        )

        # --- Step 1: Convert to Markdown ---
        if file_ext == ".pdf":
            markdown, tables = convert_pdf_to_markdown(
                input_path=absolute_path,
                table_context_window=self.table_context_window,
                file_id=asset.asset_key,
            )
        elif file_ext == ".md":
            markdown = Path(absolute_path).read_text(encoding="utf-8")
            tables = []
        else:
            raise ValueError(
                f"MarkdownProcessor does not support extension '{file_ext}' "
                f"(asset_id={asset.id}, uri={asset.source_uri})"
            )

        logger.info(
            "Conversion complete",
            extra={
                "asset_id": asset.id,
                "markdown_chars": len(markdown),
                "table_count": len(tables),
            },
        )

        # --- Step 2: Summarise tables ---
        table_chunks = summerize_and_chunk_table(
            tables,
            tags=asset.tags,
            scopes=asset.scopes,
            model=self.table_summarizer,
            summerize_enabled=self.summarize_tables_enabled,
        )
        logger.info(
            "Table summarization complete",
            extra={"asset_id": asset.id, "table_chunks": len(table_chunks)},
        )

        # --- Step 3: Chunk the markdown ---
        text_chunks = chunk_markdown(
            markdown=markdown,
            source_path=absolute_path,
            tags=asset.tags,
            scopes=asset.scopes,
            config=self.config,
            file_id=asset.asset_key,
        )
        logger.info(
            "Chunked markdown into parent chunks",
            extra={"asset_id": asset.id, "parent_chunks": len(text_chunks)},
        )

        # --- Step 4: Merge and stamp asset provenance ---
        text_chunks.extend(table_chunks)

        for chunk in text_chunks:
            chunk.asset_id = asset.id

        logger.info(
            "Processing complete",
            extra={
                "asset_id": asset.id,
                "total_chunks": len(text_chunks),
            },
        )
        return text_chunks
