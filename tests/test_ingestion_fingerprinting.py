from rag_core.ingestion.fingerprinting import generate_asset_key


def test_generate_file_id_is_semantic_and_sanitized() -> None:
    source_uri = "file:///Users/me/source docs/Finance/Quarterly Report (Final).PDF"
    file_id = generate_asset_key(source_uri)

    assert file_id.startswith("finance_quarterly_report_final_")
    assert file_id.endswith(".pdf")
    assert " " not in file_id
    assert "(" not in file_id
    assert ")" not in file_id


def test_generate_file_id_uses_8_char_hash_by_default() -> None:
    file_id = generate_asset_key("data/source_documents/legal/policy.md")
    hash_part = file_id.removesuffix(".md").rsplit("_", 1)[-1]
    assert len(hash_part) == 8


def test_generate_file_id_is_deterministic_for_equivalent_paths() -> None:
    a = generate_asset_key("data/source_documents/legal/../legal/policy.md")
    b = generate_asset_key("data/source_documents/legal/policy.md")
    assert a == b
