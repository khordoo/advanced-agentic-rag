#!/usr/bin/env python3
"""
04 — Full Pipeline (End-to-End)
===============================

Runs the complete Advanced RAG pipeline on a sample PDF:

1. PDF → numbered Markdown  (converter)
2. Markdown → parent/child chunks (chunker)
3. Store chunks  (in-memory vector store)
4. Query → hierarchical retrieval + reranking → LLM answer

**Requires** a ``.env`` file with LLM provider config (see ``.env.example``).
"""

import logging
import sys
from pathlib import Path

# Add project root to path so rag_core is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from rag_core.chunker import chunk_markdown, store_chunks
from rag_core.config import load_config
from rag_core.converter import convert_pdf_to_markdown
from rag_core.retriever import Retriever
from rag_core.pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_PDF = Path(__file__).parent / "sample_docs" / "demo.pdf"


def main():
    if not SAMPLE_PDF.is_file():
        print(f"⚠  Sample PDF not found at {SAMPLE_PDF}")
        print("   Copy a small PDF to examples/sample_docs/demo.pdf to try this.")
        return

    cfg = load_config()

    # ── Step 1: Convert PDF ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 1: Converting PDF to Markdown")
    print("=" * 70)
    markdown, tables = convert_pdf_to_markdown(SAMPLE_PDF)
    print(f"  ✓ Converted — {len(markdown)} chars, {len(tables)} tables detected")

    # ── Step 2: Chunk ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2: Chunking into parent-child hierarchy")
    print("=" * 70)
    chunks = chunk_markdown(markdown, source_path=str(SAMPLE_PDF), config=cfg)
    total_children = sum(len(c.children) for c in chunks)
    print(f"  ✓ {len(chunks)} parent chunks, {total_children} child chunks")

    # ── Step 3: Store (in-memory) ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 3: Storing in in-memory vector store")
    print("=" * 70)
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = InMemoryVectorStore(emb)
    doc_store: dict = {}
    store_chunks(chunks, vector_store, doc_store)
    print(f"  ✓ {len(doc_store)} parent docs, all children embedded")

    # ── Step 4: Query ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 4: RAG Query")
    print("=" * 70)

    retriever = Retriever(vector_store=vector_store, doc_store=doc_store, config=cfg)
    pipeline = RAGPipeline(cfg, provider="aws", retriever=retriever)

    query = "What are the main risk factors mentioned in this document?"
    print(f'  Query: "{query}"')

    result = pipeline.run(query)

    print(f"\n--- Answer ---")
    print(result["response"])
    print(f"\n  ({len(result['contexts'])} context chunks used)")


if __name__ == "__main__":
    main()
