"""Quick smoke tests for the new Markdown parser."""

from rag_core.genetation.parsers import parse_agent_response, parse_markdown_response


def test_markdown_with_citations():
    md = (
        "Higher pricing of 4% was the specific driver [1]. "
        "Foreign exchange was unfavorable at 2% [2].\n\n"
        "### Sources\n"
        "[1] report.pdf (Page 5)\n"
        "[2] annual.pdf (Page 12)"
    )
    result = parse_markdown_response(md)
    print("Test 1 (Markdown + citations):")
    print(result.model_dump_json(indent=2))
    assert len(result.citations) == 2
    assert result.citations[0].page_number == 5
    assert result.citations[0].file_name == "report.pdf"
    assert result.citations[1].file_name == "annual.pdf"
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
    md = "The data shows growth [1].\n\n### Sources\n[1] data.csv"
    result = parse_markdown_response(md)
    print("Test 3 (No page number):")
    print(result.model_dump_json(indent=2))
    assert len(result.citations) == 1
    assert result.citations[0].page_number == 0
    assert result.citations[0].file_name == "data.csv"
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
    md = "```\nAnswer text [1].\n\n### Sources\n[1] file.pdf (Page 3)\n```"
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
        "[1] report.pdf (Page 5)\n"
        "[2] data.csv\n"
        "[3] summary.pdf (Page 10)"
    )
    result = parse_markdown_response(md)
    print("Test 6 (Mixed with/without pages):")
    print(result.model_dump_json(indent=2))
    assert len(result.citations) == 3
    assert result.citations[0].page_number == 5
    assert result.citations[1].page_number == 0  # no page → 0
    assert result.citations[2].page_number == 10
    print("  PASSED\n")


if __name__ == "__main__":
    test_markdown_with_citations()
    test_no_sources()
    test_source_without_page()
    test_json_fallback()
    test_code_fenced_markdown()
    test_mixed_sources()
    print("✅ All tests passed!")
