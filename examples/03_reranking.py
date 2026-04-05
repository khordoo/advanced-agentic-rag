#!/usr/bin/env python3
"""
03 — Cross-Encoder Reranking
=============================

Demonstrates how a **cross-encoder reranker** improves retrieval quality.

Bi-encoder embeddings are fast but can miss nuance. A cross-encoder scores
each (query, document) pair jointly — slower but far more accurate. We use
it as a second stage: retrieve many candidates cheaply, then rerank the top-K.

This script shows the *before* and *after* ordering side by side.
"""

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

# ── Documents ──────────────────────────────────────────────────────────────

DOCS = [
    Document(page_content="The company's total revenue was $4.2 billion in fiscal year 2024."),
    Document(page_content="Employee satisfaction scores improved by 12% compared to the prior year."),
    Document(page_content="R&D spending increased to $680 million, focused on AI and cloud platforms."),
    Document(page_content="Revenue grew 15% year-over-year, driven by strong demand across all segments."),
    Document(page_content="The company opened three new distribution centres in Southeast Asia."),
    Document(page_content="Free cash flow was $890 million, supporting dividends and buybacks."),
    Document(page_content="Cybersecurity investment reached $120 million, a 35% increase."),
    Document(page_content="Operating margin expanded 180 basis points to 42.3%."),
]

QUERY = "What was the company's revenue growth?"

# ── Build in-memory index ──────────────────────────────────────────────────

emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
store = InMemoryVectorStore(emb)
store.add_documents(DOCS)

# ── Step 1: Bi-encoder retrieval (fast) ────────────────────────────────────

results = store.similarity_search(QUERY, k=5)

print("=" * 70)
print(f'QUERY: "{QUERY}"')
print("=" * 70)
print("\n--- BEFORE reranking (bi-encoder similarity order) ---\n")
for i, doc in enumerate(results, 1):
    print(f"  {i}. {doc.page_content}")

# ── Step 2: Cross-encoder reranking (accurate) ────────────────────────────

reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
pairs = [(QUERY, doc.page_content) for doc in results]
scores = reranker.predict(pairs)

scored = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)

print("\n--- AFTER reranking (cross-encoder score order) ---\n")
for i, (score, doc) in enumerate(scored, 1):
    print(f"  {i}. [score: {score:.4f}] {doc.page_content}")

# ── Comparison ─────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)

bi_top = results[0].page_content[:60]
ce_top = scored[0][1].page_content[:60]

if bi_top != ce_top:
    print(f"\n  ⚡ Reranker CHANGED the top result!")
    print(f"     Bi-encoder #1: \"{bi_top}…\"")
    print(f"     Reranker  #1: \"{ce_top}…\"")
else:
    print(f"\n  ✓ Same top result, but positions of other docs may have shifted.")

print(
    "\nKEY TAKEAWAY:\n"
    "  Bi-encoders are fast but approximate. Cross-encoders jointly attend to\n"
    "  the query AND document, catching semantic nuances that bi-encoders miss.\n"
    "  In production, retrieve K=15 candidates cheaply, then rerank to top-3."
)
