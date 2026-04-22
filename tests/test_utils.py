"""Tests for file identity utilities in rag_core.utils."""

import os
import tempfile
from pathlib import Path

import pytest

from rag_core.utils import (
    _sanitize_identifier,
    compute_content_hash,
    generate_file_id,
)


# ---------------------------------------------------------------------------
# _sanitize_identifier
# ---------------------------------------------------------------------------


class TestSanitizeIdentifier:
    """Verify the raw-string → agent-friendly-id cleaning rules."""

    def test_replaces_slashes(self):
        assert _sanitize_identifier("q3/earnings") == "q3_earnings"

    def test_replaces_backslashes(self):
        assert _sanitize_identifier("q3\\earnings") == "q3_earnings"

    def test_replaces_spaces(self):
        assert _sanitize_identifier("annual report") == "annual_report"

    def test_replaces_hyphens(self):
        assert _sanitize_identifier("annual-report") == "annual_report"

    def test_removes_parentheses(self):
        assert _sanitize_identifier("report (draft)") == "report_draft"

    def test_removes_brackets(self):
        assert _sanitize_identifier("report [v2]") == "report_v2"

    def test_collapses_multiple_underscores(self):
        assert _sanitize_identifier("a___b") == "a_b"

    def test_strips_leading_trailing_underscores(self):
        assert _sanitize_identifier("_hello_") == "hello"

    def test_combined_special_chars(self):
        result = _sanitize_identifier("Q3 (Earnings)/Report--Final [v2]")
        assert result == "Q3_Earnings_Report_Final_v2"

    def test_empty_string(self):
        assert _sanitize_identifier("") == ""

    def test_plain_string_unchanged(self):
        assert _sanitize_identifier("simple_name_123") == "simple_name_123"


# ---------------------------------------------------------------------------
# compute_content_hash
# ---------------------------------------------------------------------------


class TestComputeContentHash:
    """Verify SHA-256 content hashing."""

    def test_returns_hex_digest(self, tmp_path: Path):
        f = tmp_path / "a.txt"
        f.write_text("hello world")
        h = compute_content_hash(f)
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex length

    def test_same_content_same_hash(self, tmp_path: Path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("identical content")
        f2.write_text("identical content")
        assert compute_content_hash(f1) == compute_content_hash(f2)

    def test_different_content_different_hash(self, tmp_path: Path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("content A")
        f2.write_text("content B")
        assert compute_content_hash(f1) != compute_content_hash(f2)


# ---------------------------------------------------------------------------
# generate_file_id
# ---------------------------------------------------------------------------


class TestGenerateFileId:
    """Verify the full file_id generation pipeline."""

    def _make_file(self, path: Path, content: str = "test") -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def test_basic_format(self, tmp_path: Path):
        """file_id has format {folder}_{stem}_{NNN}.{ext}"""
        base = tmp_path / "docs"
        f = self._make_file(base / "earnings" / "report.pdf")
        fid = generate_file_id(f, base)

        assert fid.startswith("earnings_report_")
        assert fid.endswith(".pdf")
        # Salt is 3 digits before the extension
        stem = fid.removesuffix(".pdf")
        salt_part = stem.rsplit("_", 1)[-1]
        assert salt_part.isdigit()
        assert len(salt_part) == 3

    def test_deterministic(self, tmp_path: Path):
        """Same file → same file_id (salt is content-hash-derived)."""
        base = tmp_path / "docs"
        f = self._make_file(base / "report.pdf", content="stable content")
        fid1 = generate_file_id(f, base)
        fid2 = generate_file_id(f, base)
        assert fid1 == fid2

    def test_different_content_different_salt(self, tmp_path: Path):
        """Different content → different salt → different file_id."""
        base = tmp_path / "docs"
        f = self._make_file(base / "report.pdf", content="version A")
        fid_a = generate_file_id(f, base)

        # Overwrite with different content
        f.write_text("version B")
        fid_b = generate_file_id(f, base)

        assert fid_a != fid_b

    def test_different_folders_different_ids(self, tmp_path: Path):
        """Same filename in different folders → different file_ids."""
        base = tmp_path / "docs"
        f1 = self._make_file(base / "q3" / "report.pdf", content="q3 data")
        f2 = self._make_file(base / "q4" / "report.pdf", content="q4 data")

        fid1 = generate_file_id(f1, base)
        fid2 = generate_file_id(f2, base)

        assert fid1 != fid2
        assert fid1.startswith("q3_report_")
        assert fid2.startswith("q4_report_")

    def test_file_in_base_dir_uses_base_name(self, tmp_path: Path):
        """File directly in base_dir → parent_folder = base_dir name."""
        base = tmp_path / "my_docs"
        f = self._make_file(base / "report.pdf")
        fid = generate_file_id(f, base)
        assert fid.startswith("my_docs_report_")

    def test_special_chars_sanitized(self, tmp_path: Path):
        """Parentheses, spaces, etc. are stripped."""
        base = tmp_path / "docs"
        f = self._make_file(base / "Annual Report (Draft).pdf")
        fid = generate_file_id(f, base)

        assert "(" not in fid
        assert ")" not in fid
        assert " " not in fid
        assert fid.endswith(".pdf")
        # Should contain "Annual_Report_Draft"
        assert "Annual_Report_Draft" in fid

    def test_preserves_extension(self, tmp_path: Path):
        """Non-PDF extensions are preserved."""
        base = tmp_path / "docs"
        f = self._make_file(base / "data.md")
        fid = generate_file_id(f, base)
        assert fid.endswith(".md")
