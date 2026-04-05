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

logger = logging.getLogger(__name__)

# Default header hierarchy for markdown splitting
HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]
HEADER_KEYS = [v for _, v in HEADERS_TO_SPLIT_ON]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ParentChunk:
    """A parent chunk with its child documents ready for storage."""

    parent_id: str
    enriched_text: str
    metadata: dict[str, Any]
    children: list[Document] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pure chunking logic (no side-effects)
# ---------------------------------------------------------------------------


def chunk_markdown(
    markdown: str,
    source_path: str = "",
    config: dict | None = None,
) -> list[ParentChunk]:
    """Split numbered markdown into parent/child chunks.

    Parameters
    ----------
    markdown:
        Page-numbered markdown (output of :func:`rag_core.converter.convert_pdf_to_markdown`).
    source_path:
        Original file path — used for metadata.
    config:
        Config dict with chunk-size settings.

    Returns
    -------
    list[ParentChunk]
        Structured chunks ready to be stored.
    """
    config = config or {}
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
            child_docs = [
                Document(
                    page_content=header + child_text,
                    metadata={"parent": parent_id},
                )
                for child_text in child_texts
            ]

            all_parent_chunks.append(
                ParentChunk(
                    parent_id=parent_id,
                    enriched_text=enriched_parent,
                    metadata=parent_doc.metadata,
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


def store_chunks(
    chunks: list[ParentChunk],
    vector_store,
    doc_store: dict[str, dict],
    persist_path: str | Path | None = None,
) -> None:
    """Store parent/child chunks into a vector store and document store.

    Parameters
    ----------
    chunks:
        Output of :func:`chunk_markdown`.
    vector_store:
        Any LangChain-compatible vector store (``InMemoryVectorStore``,
        ``Chroma``, ``QdrantVectorStore``, …).
    doc_store:
        A plain ``dict`` used as the parent document store.
        Keys are parent IDs, values are ``{"content": ..., "metadata": ...}``.
    persist_path:
        If given, pickle-dump *doc_store* to this path after storing.
    """
    for chunk in chunks:
        doc_store[chunk.parent_id] = {
            "content": chunk.enriched_text,
            "metadata": json.dumps(chunk.metadata, default=str),
        }
        if chunk.children:
            vector_store.add_documents(chunk.children)

    logger.info("Stored %d parent chunks in document store", len(chunks))

    if persist_path is not None:
        persist_path = Path(persist_path)
        persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(persist_path, "wb") as fh:
            pickle.dump(doc_store, fh)
        logger.info("Document store persisted to %s", persist_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_clean_name(path: str | Path) -> str:
    return os.path.basename(str(path))
