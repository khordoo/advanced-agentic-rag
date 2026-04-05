#!/usr/bin/env python3
"""
02 — Parent-Child Retrieval
===========================

Demonstrates the **parent-child chunking pattern**:

- **Child chunks** (small, ~300 chars) are embedded and searched via
  vector similarity — small text = more precise matching.
- **Parent chunks** (large, ~1500 chars) are returned as context —
  the LLM gets the broader context around the match.

Uses an in-memory vector store — no external services needed.
"""

import uuid

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Simulate parent documents ──────────────────────────────────────────────

PARENTS = [
    {
        "id": uuid.uuid4().hex,
        "section": "Financial Highlights",
        "content": (
            "Revenue grew 15% year-over-year to $4.2 billion, driven by strong "
            "demand across all business segments. Operating income improved by 22%, "
            "reflecting disciplined cost management and operational efficiency gains. "
            "The gross margin expanded by 180 basis points to 42.3%, while free cash "
            "flow generation remained robust at $890 million, supporting continued "
            "investment in R&D and strategic acquisitions. Capital expenditures totalled "
            "$450 million, primarily directed toward expanding manufacturing capacity "
            "in key growth markets and upgrading digital infrastructure. The company "
            "returned $1.2 billion to shareholders through dividends and share "
            "buybacks during the fiscal year."
        ),
    },
    {
        "id": uuid.uuid4().hex,
        "section": "Risk Factors",
        "content": (
            "The company faces risks including market volatility, supply chain "
            "disruptions, regulatory changes across multiple jurisdictions, and "
            "increasing competition from both established players and emerging "
            "startups. Cybersecurity threats continue to evolve in sophistication "
            "and frequency. The company invested $120 million in cybersecurity "
            "infrastructure during the fiscal year, representing a 35% increase. "
            "Foreign exchange fluctuations may impact international revenues, which "
            "constitute approximately 45% of total revenue. Climate-related risks "
            "are increasingly relevant to our supply chain operations."
        ),
    },
    {
        "id": uuid.uuid4().hex,
        "section": "Corporate Strategy",
        "content": (
            "The company's strategy centres on three pillars: organic growth through "
            "product innovation, selective acquisitions in adjacent markets, and "
            "margin expansion via operational excellence. R&D investment increased "
            "to $680 million, with a focus on AI-driven analytics and cloud-native "
            "platforms. The company completed two acquisitions totalling $340 million "
            "during the year. Employee headcount grew 8% to 28,500 globally."
        ),
    },
]

# ── Build parent store and child vector index ──────────────────────────────

doc_store: dict[str, dict] = {}
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)

emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = InMemoryVectorStore(emb)

print("=" * 70)
print("BUILDING PARENT-CHILD INDEX")
print("=" * 70)

all_children: list[Document] = []

for parent in PARENTS:
    doc_store[parent["id"]] = {
        "content": f"[Section: {parent['section']}]\n\n{parent['content']}",
    }

    child_texts = child_splitter.split_text(parent["content"])
    children = [
        Document(
            page_content=f"[Section: {parent['section']}]\n{text}",
            metadata={"parent": parent["id"]},
        )
        for text in child_texts
    ]
    all_children.extend(children)

    print(f"\n  Parent: {parent['section']}")
    print(f"    Content length: {len(parent['content'])} chars")
    print(f"    Child chunks  : {len(children)}")

vector_store.add_documents(all_children)
print(f"\n  Total children in vector store: {len(all_children)}")

# ── Query & Retrieve ───────────────────────────────────────────────────────

query = "How much did the company invest in cybersecurity?"

print("\n" + "=" * 70)
print(f"QUERY: \"{query}\"")
print("=" * 70)

# Step 1: Search children (small, precise)
child_results = vector_store.similarity_search(query, k=3)

print("\n--- Step 1: Child chunk matches (what the vector search finds) ---")
for i, child in enumerate(child_results, 1):
    print(f"\n  Child {i}:")
    print(f"    Parent ID: {child.metadata['parent'][:12]}…")
    print(f"    Content  : {child.page_content[:150]}…")

# Step 2: Deduplicate & fetch parents (large, contextual)
seen = set()
parent_results = []
for child in child_results:
    pid = child.metadata["parent"]
    if pid not in seen:
        seen.add(pid)
        parent_results.append(doc_store[pid])

print("\n--- Step 2: Parent documents (what the LLM receives) ---")
for i, parent in enumerate(parent_results, 1):
    print(f"\n  Parent {i}:")
    print(f"    Content: {parent['content']}")

# ── Key takeaway ───────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("KEY TAKEAWAY")
print("=" * 70)
print(
    "  The vector search matches on SMALL child chunks (precise similarity),\n"
    "  but the LLM receives the LARGE parent chunks (richer context).\n"
    "  This gives you the best of both worlds: precision + context."
)
