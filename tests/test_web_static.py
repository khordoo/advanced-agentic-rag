from pathlib import Path


def test_web_citation_footer_renders_asset_name() -> None:
    app_js = Path("rag_core/web/static/app.js").read_text(encoding="utf-8")

    assert "cit.asset_name || assetKey" in app_js
    assert "escapeHtml(assetName)" in app_js
    assert "escapeHtml(assetKey)</span>" not in app_js
