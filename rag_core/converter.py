"""PDF → numbered Markdown conversion via Docling.
TODO: "Note: If memory becomes an issue for large docs, we should re-implement this with
      a lazy generator (see git history for previous version)."
"""

from __future__ import annotations
from typing import List
import logging
import os
import re
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling_core.types.doc import DoclingDocument,DocItemLabel, TableItem, TextItem

from rag_core.utils import get_padded_sliding_window
logger = logging.getLogger(__name__)

PAGE_TMPL = "<!-- PAGE_{n} -->"
_pattern_str = re.escape(PAGE_TMPL).replace(r"\{n\}", r"(\d+)")
PAGE_PATTERN = re.compile(_pattern_str)


def convert_pdf_to_markdown(
    input_path: str | Path,
    output_path: str | Path | None = None,
    table_context_window: int = 3
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
    table_context_window:
        Number of text elements (paragraphs/headings) to capture AFTER a table. 
        (Note: One preceding element is always included automatically.) 
        For example, a value of 3 includes the 1 preceding and 3 following paragraph
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

    enriched_tables=_enrich_tables_with_surrounding_text(result.document,table_context_window)
    logger.info(f"Completed processing {len(enriched_tables)} Tables")
    return numbered_markdown,enriched_tables



def _enrich_tables_with_surrounding_text(document: DoclingDocument,table_context_window=3) -> List[str]:
    # Convert to list so we can look ahead/behind by index
    # Dcouments are genrally small for processing in memeory should be fast
    print('===========Table contex size insdie func:',table_context_window)
    items = [item for item, _ in document.iterate_items()]
    enriched_tables = []
    for i, current in enumerate(items):
        if not isinstance(current, TableItem):
            continue

        table_context = []

        # 1. Parent Context (1 item above)
        if i > 0 and isinstance(items[i-1], TextItem):
            table_context.append(items[i-1].text)

        # 2. The Table
        table_context.append(current.export_to_markdown(doc=document))

        # 3. Child Context (Greedy look-ahead for footnotes/explanation)
        # Look at the next 3 items, but stop if we hit another Table or a new Heading
        for j in range(i + 1, min(i + (table_context_window+1), len(items))):
            next_item = items[j]
            if isinstance(next_item, TableItem) or next_item.label == DocItemLabel.SECTION_HEADER:
                break # Stop if we hit another table

            if isinstance(next_item, TextItem):
                table_context.append(next_item.text)
                # Success Logic: If it starts with '(', it's likely a footnote.
                # If it doesn't, it might be the concluding paragraph.

        enriched_tables.append('\n\n'.join(table_context))
        print(enriched_tables[-1])
        print('---'*20)

    return enriched_tables

def get_clean_name(path: str | Path) -> str:
    """Return a human-readable basename without extension or underscores."""
    filename = os.path.basename(str(path))
    return filename.replace(".pdf", "").replace("_", " ")
