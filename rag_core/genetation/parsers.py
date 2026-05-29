import re
import json
import logging

from rag_core.prompts.specs.base import RagFinalResponseSchema, Citation

logger = logging.getLogger(__name__)


def extract_text_from_message(content: str | list | dict) -> str:
    """
    Provider-agnostic text extraction from LangChain message content.
    Works seamlessly across OpenAI, Anthropic, Bedrock, and local LLMs.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_blocks = []
        for block in content:
            if isinstance(block, str):
                text_blocks.append(block)
            elif isinstance(block, dict) and "text" in block:
                # Extracts text safely regardless of other provider-specific keys
                text_blocks.append(block["text"])
        return " ".join(text_blocks).strip()

    return str(content).strip()


def extract_clean_json(raw_response: str) -> dict:
    """Extracts the JSON object from a chatty micro-model response."""
    # This regex looks for the first '{' and the last '}', grabbing everything inside.
    # It completely ignores <thinking> tags, markdown, or "Here is your answer:"
    match = re.search(r"\{.*\}", raw_response, re.DOTALL)

    if not match:
        raise ValueError(
            f"Could not locate a JSON object in the model's response: {raw_response}"
        )

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Extracted JSON was invalid: {e}\nExtracted string: {match.group(0)}"
        )


# ── Regex for parsing source lines ──────────────────────────────────────
# Matches lines like:
#   [1] annual_report_a1b2c3d4.pdf (Page 5)
#   [2] q3_earnings_report_472.pdf
_SOURCE_LINE_WITH_PAGE = re.compile(
    r"\[(\d+)\]\s+(.+?)\s+\(Page\s+(\d+)\)", re.IGNORECASE
)
_SOURCE_LINE_NO_PAGE = re.compile(r"\[(\d+)\]\s+(.+?)(?:\s*$)", re.MULTILINE)
_SOURCES_FOOTER_RE = re.compile(
    r"""
    (?P<header>
        ^|\n+
    )
    [ \t]*
    (?:
        \#{1,6}[ \t]*Sources
        |
        \*\*[ \t]*Sources[ \t]*\*\*
        |
        Sources
    )
    [ \t]*
    (?:
        :?
    )
    [ \t]*
    \n+
    (?=\s*\[\d+\]\s+)
    (?P<sources>[\s\S]*)
    \Z
    """,
    re.IGNORECASE | re.MULTILINE | re.VERBOSE,
)


def _split_sources_footer(text: str) -> tuple[str, str] | None:
    """Return ``(answer, sources_block)`` for a footer-shaped Sources section.

    The header must start on its own line and be followed by at least one
    numeric citation line. This avoids truncating normal prose such as
    "revenue sources".
    """
    match = _SOURCES_FOOTER_RE.search(text)
    if not match:
        return None

    answer_text = text[: match.start()].strip()
    sources_block = match.group("sources").strip()
    return answer_text, sources_block


def parse_markdown_response(raw: str) -> RagFinalResponseSchema:
    """Parse a Markdown response with inline [N] citations and a ### Sources footer.

    Expected format:
        Some answer text with citations [1] and [2].

        ### Sources
        [1] annual_report_a1b2c3d4.pdf (Page 5)
        [2] finance_data_4f5e6d7c.csv

    Returns a RagFinalResponseSchema with the answer text and parsed asset_key citations.
    If no ### Sources section is found, returns the full text as the answer with
    an empty citations list.
    """
    # Strip code fences if the LLM wrapped the whole output
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```\w*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
        cleaned = cleaned.strip()

    footer = _split_sources_footer(cleaned)
    if footer is None:
        # No safe Sources footer found — return full text, no citations
        logger.debug("No ### Sources section found in response, returning raw answer.")
        return RagFinalResponseSchema(answer=cleaned.strip(), citations=[])

    answer_text, sources_block = footer

    # Parse citations from the sources block
    citations: list[Citation] = []
    seen_indices: set[int] = set()

    for match in _SOURCE_LINE_WITH_PAGE.finditer(sources_block):
        idx = int(match.group(1))
        if idx not in seen_indices:
            seen_indices.add(idx)
            citations.append(
                Citation(
                    document_index=idx,
                    asset_key=match.group(2).strip(),
                    page_number=int(match.group(3)),
                )
            )

    # Second pass: pick up any source lines that didn't have a page number
    for match in _SOURCE_LINE_NO_PAGE.finditer(sources_block):
        idx = int(match.group(1))
        if idx not in seen_indices:
            seen_indices.add(idx)
            asset_key = match.group(2).strip()
            # Clean trailing parenthetical if regex caught partial junk
            asset_key = re.sub(r"\s*\(.*$", "", asset_key).strip()
            citations.append(
                Citation(
                    document_index=idx,
                    asset_key=asset_key,
                    page_number=0,  # 0 signals "unknown" to the frontend
                )
            )

    # Sort citations by document_index for consistency
    citations.sort(key=lambda c: c.document_index)

    return RagFinalResponseSchema(answer=answer_text, citations=citations)


def parse_agent_response(raw: str | list | dict) -> RagFinalResponseSchema:
    """Unified parser: detects format and dispatches to the right parser.

    Strategy:
    1. If the response looks like JSON (starts with `{`), try JSON first.
    2. Otherwise try Markdown parsing (the primary expected format).
    3. If both fail, return the raw text as the answer with no citations.
    """
    normalized = extract_text_from_message(raw)
    stripped = normalized.strip()

    # If it looks like JSON, try that first
    if stripped.startswith("{"):
        try:
            data = extract_clean_json(normalized)
            logger.debug("Parsed response as JSON.")
            parsed = RagFinalResponseSchema(**data)
            if not parsed.citations and _split_sources_footer(parsed.answer):
                return parse_markdown_response(parsed.answer)
            return parsed
        except Exception as e:
            logger.debug("JSON-like input but parsing failed: %s, trying Markdown.", e)

    # Primary path: Markdown with ### Sources
    try:
        result = parse_markdown_response(normalized)
        if result.answer:
            logger.debug("Successfully parsed Markdown response.")
            return result
    except Exception as e:
        logger.debug("Markdown parsing failed: %s", e)

    # Last resort: return the raw text as the answer with no citations
    logger.warning("All parsers failed. Returning raw response as answer.")
    return RagFinalResponseSchema(answer=stripped, citations=[])
