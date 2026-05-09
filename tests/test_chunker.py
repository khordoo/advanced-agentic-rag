"""Smoke tests for the chunker module."""

from rag_core.chunker import chunk_markdown, summerize_and_chunk_table


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
# asset_key propagation tests
# ---------------------------------------------------------------------------


_SIMPLE_MD = (
    "# Title\n\n"
    + "This is a paragraph. " * 30  # ~600 chars — enough for one parent
    + "\n\n## Section\n\n"
    + "Another paragraph. " * 30
)


def test_asset_key_propagated_to_parent_metadata():
    """When file_id is provided, parent metadata.asset_key should equal it."""
    asset_key = "q3_earnings_report_472.pdf"
    chunks = chunk_markdown(_SIMPLE_MD, source_path="report.pdf", file_id=asset_key)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.metadata.asset_key == asset_key
        assert chunk.metadata.file_name == "report.pdf"


def test_asset_key_propagated_to_child_metadata():
    """When file_id is provided, child metadata should carry it."""
    asset_key = "q4_earnings_report_831.pdf"
    chunks = chunk_markdown(_SIMPLE_MD, source_path="report.pdf", file_id=asset_key)
    for chunk in chunks:
        for child in chunk.children:
            assert child.metadata["asset_key"] == asset_key
            assert child.metadata["file_name"] == "report.pdf"


def test_asset_key_none_falls_back_to_basename():
    """When file_id is None, asset_key and file_name use the basename."""
    chunks = chunk_markdown(_SIMPLE_MD, source_path="/some/path/report.pdf")
    for chunk in chunks:
        assert chunk.metadata.asset_key == "report.pdf"
        assert chunk.metadata.file_name == "report.pdf"


def test_file_name_appears_in_enriched_text():
    """The [Doc: ...] header in enriched text should contain the display name."""
    chunks = chunk_markdown(
        _SIMPLE_MD,
        source_path="/some/path/report.pdf",
        file_id="earnings_annual_report_123.pdf",
    )
    for chunk in chunks:
        assert "[Doc: report.pdf]" in chunk.enriched_text


def test_table_chunks_carry_asset_key_and_display_file_name():
    table_chunks = summerize_and_chunk_table(
        [
            {
                "asset_key": "annual_report_a1b2c3d4.pdf",
                "file_name": "Annual Report.pdf",
                "source_path": "/docs/Annual Report.pdf",
                "page_no": 3,
                "markdown": "| A | B |\n|---|---|\n| 1 | 2 |",
            }
        ],
        model=None,
        summerize_enabled=False,
    )

    assert len(table_chunks) == 1
    chunk = table_chunks[0]
    assert chunk.metadata.asset_key == "annual_report_a1b2c3d4.pdf"
    assert chunk.metadata.file_name == "Annual Report.pdf"
    assert chunk.children[0].metadata["asset_key"] == "annual_report_a1b2c3d4.pdf"
    assert chunk.children[0].metadata["file_name"] == "Annual Report.pdf"
