"""Smoke tests for the retriever module."""

import uuid

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from rag_core.retriever import Retriever


def _build_test_store():
    """Create a minimal in-memory store with three parent/child pairs."""
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = InMemoryVectorStore(emb)
    doc_store = {}

    for text in [
        "Revenue grew 15% to $4.2 billion in fiscal year 2024.",
        "Cybersecurity investment reached $120 million.",
        "R&D spending increased to $680 million.",
    ]:
        pid = uuid.uuid4().hex
        doc_store[pid] = {"content": text, "metadata": "{}"}
        child = Document(page_content=text, metadata={"parent": pid})
        vector_store.add_documents([child])

    return vector_store, doc_store


def test_similarity_search_returns_results():
    """Similarity search should return matching documents."""
    vs, ds = _build_test_store()
    retriever = Retriever(vector_store=vs, doc_store=ds)
    results = retriever.similarity_search("revenue growth", k=2)
    assert len(results) > 0


def test_hierarchical_search_returns_parent_content():
    """Hierarchical search should return parent dicts with content and metadata."""
    vs, ds = _build_test_store()
    retriever = Retriever(vector_store=vs, doc_store=ds)
    results = retriever.hierarchical_search("cybersecurity spending", k=3, top_n=2)
    assert len(results) > 0
    assert all(isinstance(r, dict) for r in results)
    assert all("content" in r and "metadata" in r for r in results)
