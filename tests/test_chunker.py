"""Smoke tests for the chunker module."""

from rag_core.chunker import chunk_markdown


def test_chunk_markdown_returns_parents():
    """Chunking a simple markdown doc should produce at least one parent."""
    md = "# Title\n\nSome content here that is long enough to be chunked.\n\n## Section\n\nMore content."
    chunks = chunk_markdown(md, source_path="test.pdf")
    assert len(chunks) >= 1
    assert all(c.parent_id for c in chunks)


def test_chunk_markdown_creates_children():
    """Each parent should have at least one child document."""
    md = (
        "# Title\n\n"
        + "This is a paragraph. " * 50  # ~1000 chars
        + "\n\n## Section\n\n"
        + "Another paragraph. " * 50
    )
    chunks = chunk_markdown(md, source_path="test.pdf")
    for chunk in chunks:
        assert len(chunk.children) >= 1
        # Children should reference their parent
        for child in chunk.children:
            assert child.metadata["parent"] == chunk.parent_id
