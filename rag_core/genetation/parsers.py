import re
import json


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
