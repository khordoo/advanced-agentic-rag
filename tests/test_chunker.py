"""Smoke tests for the chunker module."""

from rag_core.chunker import chunk_markdown


def test_chunk_markdown_returns_parents():
    """Chunking a simple markdown doc should produce at least one parent."""
    md = "# Title\n\nSome content here that is long enough to be chunked.\n\n## Section\n\nMore content."
    chunks = chunk_markdown(md, source_path="test.pdf")
    assert len(chunks) >= 1
    assert all(c.id for c in chunks)


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
            assert child.metadata["parent_id"] == chunk.id


# ---------------------------------------------------------------------------
# file_id propagation tests
# ---------------------------------------------------------------------------


_SIMPLE_MD = (
    "# Title\n\n"
    + "This is a paragraph. " * 30  # ~600 chars — enough for one parent
    + "\n\n## Section\n\n"
    + "Another paragraph. " * 30
)


def test_file_id_propagated_to_parent_metadata():
    """When file_id is provided, parent metadata.file_name should equal it."""
    custom_id = "q3_earnings_report_472.pdf"
    chunks = chunk_markdown(_SIMPLE_MD, source_path="report.pdf", file_id=custom_id)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.metadata.file_name == custom_id


def test_file_id_propagated_to_child_metadata():
    """When file_id is provided, child metadata should carry it."""
    custom_id = "q4_earnings_report_831.pdf"
    chunks = chunk_markdown(_SIMPLE_MD, source_path="report.pdf", file_id=custom_id)
    for chunk in chunks:
        for child in chunk.children:
            assert child.metadata["file_name"] == custom_id


def test_file_id_none_falls_back_to_basename():
    """When file_id is None, file_name should be the basename of source_path."""
    chunks = chunk_markdown(_SIMPLE_MD, source_path="/some/path/report.pdf")
    for chunk in chunks:
        assert chunk.metadata.file_name == "report.pdf"


def test_file_id_appears_in_enriched_text():
    """The [Doc: ...] header in enriched text should contain the file_id."""
    custom_id = "earnings_annual_report_123.pdf"
    chunks = chunk_markdown(_SIMPLE_MD, source_path="report.pdf", file_id=custom_id)
    for chunk in chunks:
        assert custom_id in chunk.enriched_text
