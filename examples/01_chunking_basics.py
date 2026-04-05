#!/usr/bin/env python3
"""
01 — Chunking Basics
====================

Demonstrates the two-stage chunking strategy:

1. **Markdown Header Splitting** — splits by ``#``, ``##``, ``###`` headers,
   preserving section metadata.
2. **Recursive Character Splitting** — sub-splits large sections into
   manageable chunks with configurable overlap (sliding window).

No external services required — pure text processing.
"""

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# ── Sample document ─────────────────────────────────────────────────────────

SAMPLE_MARKDOWN = """
# Introduction
Welcome to Acme Corp's annual report. This section provides a high-level overview.

## Financial Highlights
Revenue grew 15% year-over-year to $4.2 billion, driven by strong demand across
all business segments. Operating income improved by 22%, reflecting disciplined
cost management and operational efficiency gains. The gross margin expanded by
180 basis points to 42.3%, while free cash flow generation remained robust at
$890 million, supporting continued investment in R&D and strategic acquisitions.
Capital expenditures totalled $450 million, primarily directed toward expanding
manufacturing capacity in key growth markets and upgrading digital infrastructure.

## Risk Factors
The company faces risks including market volatility, supply chain disruptions,
regulatory changes across multiple jurisdictions, and increasing competition
from both established players and emerging startups in the technology sector.

### Cybersecurity Risks
Cybersecurity threats continue to evolve in sophistication and frequency.
The company invested $120 million in cybersecurity infrastructure during the
fiscal year, representing a 35% increase over the prior period. Despite these
investments, no assurance can be given that the company's systems will not be
breached, and any such breach could have material adverse effects.
"""

# ── Step 1: Split by headers ────────────────────────────────────────────────

headers = [
    ("#", "Header_1"),
    ("##", "Header_2"),
    ("###", "Header_3"),
]

md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
header_chunks = md_splitter.split_text(SAMPLE_MARKDOWN)

print("=" * 70)
print("STEP 1: Markdown Header Splitting")
print("=" * 70)
for i, chunk in enumerate(header_chunks, 1):
    print(f"\n--- Chunk {i} ---")
    print(f"  Metadata : {chunk.metadata}")
    print(f"  Length   : {len(chunk.page_content)} chars")
    print(f"  Preview  : {chunk.page_content[:120]}...")
print(f"\n  Total chunks after header split: {len(header_chunks)}")

# ── Step 2: Recursive sub-splitting ─────────────────────────────────────────

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30,  # 30-char sliding window overlap
)

final_chunks = recursive_splitter.split_documents(header_chunks)

print("\n" + "=" * 70)
print("STEP 2: Recursive Character Splitting (chunk_size=200, overlap=30)")
print("=" * 70)
for i, chunk in enumerate(final_chunks, 1):
    print(f"\n--- Chunk {i} ---")
    print(f"  Metadata: {chunk.metadata}")
    print(f"  Length  : {len(chunk.page_content)} chars")
    print(f"  Content : {chunk.page_content}")

print(f"\n  Total chunks after recursive split: {len(final_chunks)}")
print(
    f"\n  Summary: {len(header_chunks)} header sections → "
    f"{len(final_chunks)} final chunks"
)

# ── Key takeaway ────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("KEY TAKEAWAY")
print("=" * 70)
print(
    "  Header splitting preserves document structure (which section is this from?).\n"
    "  Recursive splitting ensures no chunk is too large for the embedding model,\n"
    "  while the overlap preserves context at chunk boundaries."
)
