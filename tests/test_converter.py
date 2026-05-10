from pathlib import Path
from types import SimpleNamespace

from rag_core.ingestion.processing import converter as converter_module
from rag_core.ingestion.processing.converter import convert_pdf_to_markdown


class _FakeDocument:
    def export_to_markdown(self, *, page_break_placeholder: str) -> str:
        return f"{page_break_placeholder}\n# Report"


class _FakeConverter:
    def convert(self, input_path: str) -> SimpleNamespace:
        return SimpleNamespace(document=_FakeDocument())


def test_convert_pdf_to_markdown_passes_asset_name_to_table_enrichment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_path = tmp_path / "Annual Report.pdf"
    source_path.write_text("fake pdf bytes", encoding="utf-8")

    monkeypatch.setattr(
        converter_module,
        "DocumentConverter",
        lambda: _FakeConverter(),
    )
    monkeypatch.setattr(
        converter_module,
        "_remove_tables_from_markdown",
        lambda raw_markdown, doc: raw_markdown,
    )
    monkeypatch.setattr(
        converter_module,
        "_add_page_number_to_markdown",
        lambda raw_markdown, page_template: "numbered markdown",
    )

    def fake_enrich(document, source_path, table_context_window, file_id, asset_name):
        return [
            {
                "asset_key": file_id,
                "asset_name": asset_name,
                "source_path": str(source_path),
                "page_no": 1,
                "markdown": "| A |\n|---|",
            }
        ]

    monkeypatch.setattr(
        converter_module,
        "_enrich_tables_with_surrounding_text",
        fake_enrich,
    )

    markdown, tables = convert_pdf_to_markdown(
        source_path,
        file_id="annual_report_a1b2c3d4.pdf",
        asset_name="Annual Report.pdf",
    )

    assert markdown == "numbered markdown"
    assert tables[0]["asset_key"] == "annual_report_a1b2c3d4.pdf"
    assert tables[0]["asset_name"] == "Annual Report.pdf"
    assert "file_name" not in tables[0]
