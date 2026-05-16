import json
from typing import Any


class ChatPresenter:
    """
    Transforms raw agent dictionary outputs into client-specific formats.
    """

    @staticmethod
    def to_markdown(agent_output: dict[str, Any]) -> str:
        final_text = agent_output.get("answer", "")
        citations = agent_output.get("citations", [])

        if citations:
            final_text += "\n\n---\n**Sources:**\n"
            for idx, cite in enumerate(citations):
                # TODO: asset_name is empty for soem asset investiage. temporarily we use asset key in its place
                doc_name = cite.get("asset_key") or cite.get("asset_key", "Unknown Document")
                page = cite.get("page_number", "?")
                source_url = cite.get("source_url", "")
                if source_url:
                    # If it's a local path, we prepend 'file://' so the browser recognizes it
                    href = f"file://{source_url}" if source_url.startswith("/") else source_url
                    final_text += f"* [{idx + 1}] [{doc_name} - Page {page}]({href})\n"
                else:
                    # Fallback if no URL exists
                    final_text += f"* [{idx + 1}] {doc_name} - Page {page}\n"

        return final_text

    @staticmethod
    def to_json_string(agent_output: dict[str, Any]) -> str:
        # Returns the raw dictionary as a formatted JSON string
        return json.dumps(agent_output, indent=2)
