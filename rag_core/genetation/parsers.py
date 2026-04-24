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
#   [1] report.pdf (Page 5)
#   [2] Q3_Earnings_Report.pdf
_SOURCE_LINE_WITH_PAGE = re.compile(
    r"\[(\d+)\]\s+(.+?)\s+\(Page\s+(\d+)\)", re.IGNORECASE
)
_SOURCE_LINE_NO_PAGE = re.compile(r"\[(\d+)\]\s+(.+?)(?:\s*$)", re.MULTILINE)


def parse_markdown_response(raw: str) -> RagFinalResponseSchema:
    """Parse a Markdown response with inline [N] citations and a ### Sources footer.

    Expected format:
        Some answer text with citations [1] and [2].

        ### Sources
        [1] report.pdf (Page 5)
        [2] data.csv

    Returns a RagFinalResponseSchema with the answer text and parsed citations.
    If no ### Sources section is found, returns the full text as the answer with
    an empty citations list.
    """
    # Strip code fences if the LLM wrapped the whole output
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```\w*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
        cleaned = cleaned.strip()

    # Split on the LAST occurrence of "### Sources" (case-insensitive)
    # Using the last one guards against the LLM echoing the example section.
    parts = re.split(r"###\s+Sources", cleaned, flags=re.IGNORECASE)

    if len(parts) <= 1:
        # No ### Sources section found — return full text, no citations
        logger.debug("No ### Sources section found in response, returning raw answer.")
        return RagFinalResponseSchema(answer=cleaned.strip(), citations=[])

    answer_text = "### Sources".join(parts[:-1]).strip()
    sources_block = parts[-1]

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
                    file_name=match.group(2).strip(),
                    page_number=int(match.group(3)),
                )
            )

    # Second pass: pick up any source lines that didn't have a page number
    for match in _SOURCE_LINE_NO_PAGE.finditer(sources_block):
        idx = int(match.group(1))
        if idx not in seen_indices:
            seen_indices.add(idx)
            file_name = match.group(2).strip()
            # Clean trailing parenthetical if regex caught partial junk
            file_name = re.sub(r"\s*\(.*$", "", file_name).strip()
            citations.append(
                Citation(
                    document_index=idx,
                    file_name=file_name,
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
            return RagFinalResponseSchema(**data)
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
