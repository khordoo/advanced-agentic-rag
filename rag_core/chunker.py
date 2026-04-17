"""Markdown → parent / child chunk splitting and storage.

Uses a 3-stage chunking strategy:

  Stage 1 — **Header split**: Split the markdown along its natural heading
  boundaries (#, ##, ###). This preserves the document's logical structure
  so a section about "Risk Factors" stays together with its metadata. A naive
  character splitter would cut right through a section, losing that context.

  Stage 2 — **Recursive split into parents**: A single header section can be
  thousands of characters (e.g. "Financial Highlights" spanning 3 pages).
  We recursively split each section into manageable parent chunks (~1,500 ch)
  that inherit the section metadata. These are what the LLM reads.

  Stage 3 — **Sub-split into children**: For vector search, smaller text
  produces more focused embeddings. Each parent is split into child chunks
  (~300 ch) that get embedded in the vector DB. Each child keeps a pointer
  back to its parent for lookup at query time.

This module is **backend-agnostic**: it accepts any LangChain-compatible
vector store so you can switch between InMemoryVectorStore (learning track),
Chroma, or Qdrant (production track) without changing the chunking logic.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from rag_core.converter import PAGE_PATTERN
from rag_core.schemas import ParentChunk, BaseChunkMetadata, ChildMetadata, ParentMetadata
from rag_core.agents import TableSummarizer
from rag_core.indexing.qdrant_indexer import QdrantIndexer
logger = logging.getLogger(__name__)

# Default header hierarchy for markdown splitting
HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]
HEADER_KEYS = [v for _, v in HEADERS_TO_SPLIT_ON]





# ---------------------------------------------------------------------------
# Pure chunking logic (no side-effects)
# ---------------------------------------------------------------------------


def chunk_markdown(
    markdown: str,
    source_path: str = "",
    tags: list[str] | None = None,
    scopes: list[str] | None = None,
    config: dict | None = None,
) -> list[ParentChunk]:
    """Split numbered markdown into parent/child chunks.

    Parameters
    ----------
    markdown:
        Page-numbered markdown (output of :func:`rag_core.converter.convert_pdf_to_markdown`).
    source_path:
        Original file path — used for metadata.
    tags: list[str] | None
        General purouse tags. e.g fincnce, legal, source like Sharedpoint vs blob etc
    scopes: list[str] | None
         RBAC filters, Assigns security/access control scopes for a file. example ['restricted']
    config:
        Config dict with chunk-size settings.

    Returns
    -------
    list[ParentChunk]
        Structured chunks ready to be stored.
    """
    config = config or {}
    if tags is None:
        tags = []
    if scopes is None:
        scopes = []
    parent_chunk_size = int(config.get("PARENT_CHUNK_SIZE", 1500))
    parent_chunk_overlap = int(config.get("PARENT_CHUNK_OVERLAP", 150))
    child_chunk_size = int(config.get("CHILD_CHUNK_SIZE", 300))
    child_chunk_overlap = int(config.get("CHILD_CHUNK_OVERLAP", 50))

    file_name = _get_clean_name(source_path)

    # --- Phase 1: split by markdown headers --------------------------------
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
    )
    header_splits = md_splitter.split_text(markdown)

    # Attach page metadata from <!-- PAGE_N --> markers
    last_seen_page = 1
    for doc in header_splits:
        pages = [int(x) for x in PAGE_PATTERN.findall(doc.page_content)]
        if pages:
            doc.metadata["page_start"] = min(pages)
            doc.metadata["page_end"] = max(pages)
            doc.metadata["pages"] = sorted(set(pages))
            last_seen_page = doc.metadata["page_end"]
        else:
            doc.metadata["page_start"] = last_seen_page
            doc.metadata["page_end"] = last_seen_page
            doc.metadata["pages"] = [last_seen_page]

        # Strip page markers from text (metadata already captured)
        doc.page_content = PAGE_PATTERN.sub("", doc.page_content).strip()
        doc.metadata["source_path"] = source_path
        doc.metadata["file_name"] = file_name
        doc.metadata["tags"] = tags
        doc.metadata["scopes"] = scopes

    # --- Phase 2: recursive split into parent chunks -----------------------
    parent_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_chunk_overlap,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap,
    )

    all_parent_chunks: list[ParentChunk] = []

    for header_doc in header_splits:
        parent_docs = parent_splitter.split_documents([header_doc])

        for parent_doc in parent_docs:
            parent_id = uuid.uuid4().hex

            breadcrumb = " > ".join(
                str(parent_doc.metadata[k])
                for k in HEADER_KEYS
                if k in parent_doc.metadata
            )
            page_start = parent_doc.metadata.get("page_start")
            page_end = parent_doc.metadata.get("page_end")
            page_label = (
                f"Page {page_start}"
                if page_start == page_end
                else f"Pages {page_start}-{page_end}"
            )

            header = (
                f"[Doc: {parent_doc.metadata.get('file_name')}] "
                f"[Section: {breadcrumb}] "
                f"[Location: {page_label}]\n\n"
            )
            enriched_parent = header + parent_doc.page_content

            # --- Phase 3: split parent into children -----------------------
            child_texts = child_splitter.split_text(parent_doc.page_content)
            child_meta=_create_child_metadata(parent_doc,parent_id).model_dump()
            child_docs = [
                Document(
                    page_content=header + child_text,
                    metadata=child_meta,
                )
                for child_text in child_texts
            ]
            all_parent_chunks.append(
                ParentChunk(
                    id=parent_id,
                    enriched_text=enriched_parent,
                    metadata=_create_parent_metadata(parent_doc),
                    children=child_docs,
                )
            )

    logger.info(
        "Chunked into %d parent chunks with %d total children",
        len(all_parent_chunks),
        sum(len(p.children) for p in all_parent_chunks),
    )
    return all_parent_chunks

# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

# Map from schema field name → the key used inside Document.metadata.
# Only add entries here when the metadata key differs from the schema field.
_META_KEY_ALIASES: dict[str, str] = {
    "source": "source_path",
}


def _extract_base_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Pull exactly the fields declared on BaseChunkMetadata from *meta*.

    Key aliases (e.g. schema 'source' ← doc 'source_path') are resolved
    automatically via ``_META_KEY_ALIASES``.
    """
    result: dict[str, Any] = {}
    for field_name, field_info in BaseChunkMetadata.model_fields.items():
        lookup_key = _META_KEY_ALIASES.get(field_name, field_name)
        default = field_info.default if field_info.default is not None else (
            field_info.default_factory() if field_info.default_factory else None
        )
        result[field_name] = meta.get(lookup_key, default)
    return result


def _create_child_metadata(parent_doc: Document, parent_id: str) -> ChildMetadata:
    base = _extract_base_metadata(parent_doc.metadata)
    return ChildMetadata(**base, parent_id=parent_id)


def _create_parent_metadata(doc: Document) -> ParentMetadata:
    base = _extract_base_metadata(doc.metadata)
    # Forward any header keys (h1, h2, h3…) as extra fields
    for header in HEADER_KEYS:
        if header_val := doc.metadata.get(header):
            base[header] = header_val
    return ParentMetadata(**base)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def summerize_and_chunk_table(table_data_list: list[dict], model:TableSummarizer,tags:list[str] |None=None,scopes: list[str] |None =None,  summerize_enabled=True) -> list[ParentChunk]:
    logger.info("Summarizing %d tables", len(table_data_list))
    if tags is None:
        tags = []
    if scopes is None:
        scopes = []

    parent_chunks=[]
    for i,table_data in enumerate(table_data_list):
        table_id = f"DATA_TABLE_{i:03d}"
        raw_md = table_data['markdown']
        base_header = f"[Doc: {table_data['file_name']}] [Section: Data Table] [Location: Page {table_data['page_no']}]"

        logger.debug(
            "Summarizing table %s from %s page %s",
            table_id,
            table_data['file_name'],
            table_data['page_no'],
        )
        table_summary =model.summarize(table_markdown=raw_md) if summerize_enabled else table_data
        logger.debug(f"Summary ready for table {table_id}. Progress: {(i+1)/len(table_data_list):0.2f}" )
        parent_id = uuid.uuid4().hex
        # Create the Child (The Searchable Vector)
        # We explicitly put the table_id in the text so the vector DB indexes it!
        child_text = f"{base_header}[SUMMARY FOR {table_id}]\n\n{table_summary}"

        # Build base metadata from the table dict, reusing the same
        # schema-driven extraction that text chunks use.
        table_meta_raw = {
            "source_path": table_data['source_path'],
            "file_name":   table_data['file_name'],
            "pages":       [table_data['page_no']],
            "tags":        tags,
            "scopes":      scopes,
        }
        base = _extract_base_metadata(table_meta_raw)

        child_doc = Document(
            page_content=child_text,
            metadata=ChildMetadata(**base, parent_id=parent_id).model_dump(),
        )

        parent_metadata = {**base, "table_id": table_id}
        # Create the Parent (The LLM Payload)
        # We append the ID to the top so the LLM has a visual anchor
        parent_text = f"{base_header}\n[{table_id}]\n\n{raw_md}"
        parent_chunks.append(
            ParentChunk(
                id=parent_id,
                enriched_text=parent_text,
                metadata=ParentMetadata(**parent_metadata),
                children=[child_doc],
            )
        )

    logger.info("Created %d table parent chunks", len(parent_chunks))
    return parent_chunks

def _get_clean_name(path: str | Path) -> str:
    return os.path.basename(str(path))
