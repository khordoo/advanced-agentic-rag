"""PDF → numbered Markdown conversion via Docling."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling_core.types.doc import TableItem, TextItem
logger = logging.getLogger(__name__)

PAGE_TMPL = "<!-- PAGE_{n} -->"
_pattern_str = re.escape(PAGE_TMPL).replace(r"\{n\}", r"(\d+)")
PAGE_PATTERN = re.compile(_pattern_str)

def _convert_tables_to_markdown(document: Document) -> str:
    """Convert tables to markdown."""
    markdown_tables = []
    for i, element in enumerate(document.elements):
        if isinstance(TableItem):
            if i>0 and isinstance(TextItem):
               preced_text = ''
            table_markdown = element.export_to_markdown()



def convert_pdf_to_markdown(
    input_path: str | Path,
    output_path: str | Path | None = None,
) -> tuple[str, list]:
    """Convert a PDF file to page-numbered Markdown.

    Each page is prefixed with an HTML comment ``<!-- PAGE_<n> -->`` that
    downstream chunkers can use to attach page metadata.

    Parameters
    ----------
    input_path:
        Path to the source PDF.
    output_path:
        If provided, the numbered markdown is written to this dir.

    Returns
    -------
    tuple[str, list]
        ``(numbered_markdown, tables)`` where *tables* is a list of
        Docling ``Table`` objects detected in the document.
    """
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Source PDF not found: {input_path}")

    converter = DocumentConverter()
    logger.info("Converting %s to markdown …", input_path.name)
    result = converter.convert(str(input_path))
    # print('Result:',result)
    # all_items=result.document.items()
    # print('all items:',all_items)
    # exit(1)
    # Export with page-break placeholders
    raw_markdown = result.document.export_to_markdown(
        page_break_placeholder=PAGE_TMPL
    )

    # Replace generic placeholders with numbered markers
    parts = raw_markdown.split(PAGE_TMPL)
    numbered_markdown = "\n\n".join(
        f"{PAGE_TMPL.format(n=i + 1)}\n{part.strip()}"
        for i, part in enumerate(parts)
        if part.strip()
    )
    # Persist to disk (optional)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(numbered_markdown, encoding="utf-8")
        logger.info("Markdown written to %s", output_path)

    tables = list(result.document.tables)
    logger.info("Detected %d table(s)", len(tables))

    return numbered_markdown, tables


def get_clean_name(path: str | Path) -> str:
    """Return a human-readable basename without extension or underscores."""
    filename = os.path.basename(str(path))
    return filename.replace(".pdf", "").replace("_", " ")
