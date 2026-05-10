"""Quick smoke tests for the new Markdown parser."""

from rag_core.genetation.parsers import (
    parse_agent_response,
    parse_markdown_response,
    extract_text_from_message,
)


def test_markdown_with_citations():
    md = (
        "Higher pricing of 4% was the specific driver [1]. "
        "Foreign exchange was unfavorable at 2% [2].\n\n"
        "### Sources\n"
        "[1] report_a1b2c3d4.pdf (Page 5)\n"
        "[2] annual_d4c3b2a1.pdf (Page 12)"
    )
    result = parse_markdown_response(md)
    print("Test 1 (Markdown + citations):")
    print(result.model_dump_json(indent=2))
    assert len(result.citations) == 2
    assert result.citations[0].page_number == 5
    assert result.citations[0].asset_key == "report_a1b2c3d4.pdf"
    assert result.citations[0].asset_name == ""
    assert result.citations[1].asset_key == "annual_d4c3b2a1.pdf"
    assert result.citations[1].page_number == 12
    print("  PASSED\n")


def test_no_sources():
    md = "I cannot find the answer in the provided documents."
    result = parse_markdown_response(md)
    print("Test 2 (No sources section):")
    print(result.model_dump_json(indent=2))
    assert result.citations == []
    assert "cannot find" in result.answer
    print("  PASSED\n")


def test_source_without_page():
    md = "The data shows growth [1].\n\n### Sources\n[1] data_9a8b7c6d.csv"
    result = parse_markdown_response(md)
    print("Test 3 (No page number):")
    print(result.model_dump_json(indent=2))
    assert len(result.citations) == 1
    assert result.citations[0].page_number == 0
    assert result.citations[0].asset_key == "data_9a8b7c6d.csv"
    print("  PASSED\n")


def test_json_fallback():
    json_str = '{"answer": "test answer", "citations": []}'
    result = parse_agent_response(json_str)
    print("Test 4 (JSON fallback):")
    print(result.model_dump_json(indent=2))
    assert result.answer == "test answer"
    assert result.citations == []
    print("  PASSED\n")


def test_code_fenced_markdown():
    md = "```\nAnswer text [1].\n\n### Sources\n[1] file_1a2b3c4d.pdf (Page 3)\n```"
    result = parse_markdown_response(md)
    print("Test 5 (Code-fenced):")
    print(result.model_dump_json(indent=2))
    assert len(result.citations) == 1
    assert result.citations[0].page_number == 3
    print("  PASSED\n")


def test_mixed_sources():
    md = (
        "Growth was driven by pricing [1] and volume [2]. "
        "Additional detail in [3].\n\n"
        "### Sources\n"
        "[1] report_a1b2c3d4.pdf (Page 5)\n"
        "[2] data_9a8b7c6d.csv\n"
        "[3] summary_1234abcd.pdf (Page 10)"
    )
    result = parse_markdown_response(md)
    print("Test 6 (Mixed with/without pages):")
    print(result.model_dump_json(indent=2))
    assert len(result.citations) == 3
    assert result.citations[0].page_number == 5
    assert result.citations[1].page_number == 0  # no page → 0
    assert result.citations[2].page_number == 10
    print("  PASSED\n")


def test_file_id_style_citations():
    """Parser handles asset_key values."""
    md = (
        "Revenue grew 15% [1]. Details in [2].\n\n"
        "### Sources\n"
        "[1] q3_earnings_report_472.pdf (Page 8)\n"
        "[2] q4_earnings_report_831.pdf"
    )
    result = parse_markdown_response(md)
    print("Test 7 (file_id-style citations):")
    print(result.model_dump_json(indent=2))
    assert len(result.citations) == 2
    assert result.citations[0].asset_key == "q3_earnings_report_472.pdf"
    assert result.citations[0].page_number == 8
    assert result.citations[1].asset_key == "q4_earnings_report_831.pdf"
    assert result.citations[1].page_number == 0
    print("  PASSED\n")


def test_extract_text_from_message_string_passthrough():
    content = "Simple answer."
    assert extract_text_from_message(content) == "Simple answer."


def test_extract_text_from_message_mixed_list():
    content = [
        {"type": "text", "text": "First"},
        "Second",
        {"type": "reasoning_content", "reasoning_content": {"text": "Ignore this"}},
        {"unexpected": "value"},
    ]
    assert extract_text_from_message(content) == "First Second"


def test_extract_text_from_message_dict_fallback():
    content = {"foo": "bar"}
    assert extract_text_from_message(content) == "{'foo': 'bar'}"


def test_parse_agent_response_accepts_list_content():
    content = [
        {"type": "text", "text": "I cannot find the answer in the provided documents."},
        {"type": "reasoning_content", "reasoning_content": {"text": "hidden"}},
    ]
    result = parse_agent_response(content)
    assert result.answer == "I cannot find the answer in the provided documents."
    assert result.citations == []


if __name__ == "__main__":
    test_markdown_with_citations()
    test_no_sources()
    test_source_without_page()
    test_json_fallback()
    test_code_fenced_markdown()
    test_mixed_sources()
    test_file_id_style_citations()
    test_extract_text_from_message_string_passthrough()
    test_extract_text_from_message_mixed_list()
    test_extract_text_from_message_dict_fallback()
    test_parse_agent_response_accepts_list_content()
    print("✅ All tests passed!")
