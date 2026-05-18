from app.services.formatters import ChatPresenter


def test_markdown_sources_prefer_asset_name_over_asset_key() -> None:
    content = ChatPresenter.to_markdown(
        {
            "answer": "Answer",
            "citations": [
                {
                    "asset_key": "source_documents_report_ef7a4579.pdf",
                    "asset_name": "report.pdf",
                    "page_number": 2,
                    "source_url": "/tmp/report.pdf",
                }
            ],
        }
    )

    assert "[report.pdf - Page 2](file:///tmp/report.pdf)" in content


def test_markdown_sources_fallback_to_asset_key_when_asset_name_missing() -> None:
    content = ChatPresenter.to_markdown(
        {
            "answer": "Answer",
            "citations": [{"asset_key": "asset-1", "page_number": 1}],
        }
    )

    assert "* [1] asset-1 - Page 1" in content
