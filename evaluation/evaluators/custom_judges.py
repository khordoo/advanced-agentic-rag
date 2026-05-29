"""Project-specific evaluation judges.

This module intentionally avoids importing Phoenix at module import time.
The deterministic judge below is used by both ground-truth and drift
evaluation flows without adding LLM judge cost or latency.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_INLINE_CITATION_RE = re.compile(r"(?:\[|【)(\d+)(?:\]|】)")


class SourceReferenceFormatJudge:
    """Evaluate whether inline source citations match structured metadata.

    The project final response is usually a serialized JSON payload with
    ``answer`` and ``citations`` fields. Inline citation markers in the
    answer text, such as ``[1]`` or ``【1】``, must correspond to
    ``citations[*].document_index`` values.
    """

    name = "reference_format_judge"

    def __init__(self) -> None:
        self.name = "reference_format_judge"
        self.__name__ = self.name
        self.__qualname__ = self.name

    def __repr__(self) -> str:
        return self.name

    __str__ = __repr__

    def evaluate(self, answer: str, **kwargs: Any) -> dict[str, Any]:
        """Return a binary score dict for the citation contract in *answer*."""
        answer_text = answer or ""
        provided_indices: set[int] = set()

        try:
            data = json.loads(answer_text)
        except json.JSONDecodeError:
            data = None

        if isinstance(data, dict):
            answer_text = str(data.get("answer") or "")
            citations = data.get("citations") or []
            if isinstance(citations, list):
                for citation in citations:
                    if not isinstance(citation, dict):
                        continue
                    document_index = citation.get("document_index")
                    try:
                        provided_indices.add(int(document_index))
                    except (TypeError, ValueError):
                        continue

        inline_indices = {
            int(match.group(1)) for match in _INLINE_CITATION_RE.finditer(answer_text)
        }

        if not inline_indices:
            return {"score": 1.0, "label": "no_citations_needed"}

        if not provided_indices:
            return {"score": 0.0, "label": "missing_structured_citations"}

        missing_indices = sorted(inline_indices - provided_indices)
        if missing_indices:
            logger.debug(
                "SourceReferenceFormatJudge: unmatched citations found: %s",
                missing_indices,
            )
            return {
                "score": 0.0,
                "label": f"unmatched_citations_{missing_indices}",
            }

        return {
            "score": 1.0,
            "label": "formatted_and_matched_correctly",
        }

    def __call__(
        self,
        input: dict | None = None,  # noqa: A002 - Phoenix evaluator signature
        output: dict | None = None,
        expected: dict | None = None,
        metadata: dict | None = None,
    ) -> dict[str, Any]:
        """Phoenix-compatible callable wrapper.

        Phoenix experiment evaluators receive ``input``, ``output``,
        ``expected``, and ``metadata``. Local dataframe flows can call
        :meth:`evaluate` directly.
        """
        answer = ""
        if isinstance(output, dict):
            if isinstance(output.get("agent_output"), dict):
                answer = json.dumps(output["agent_output"], ensure_ascii=False)
            elif isinstance(output.get("raw_response"), str):
                answer = output["raw_response"]
            elif "citations" in output:
                answer = json.dumps(
                    {
                        "answer": output.get("answer") or output.get("response") or "",
                        "citations": output.get("citations") or [],
                    },
                    ensure_ascii=False,
                )
            else:
                answer = str(output.get("answer") or output.get("response") or "")
        return self.evaluate(answer=answer)


# =====================================================================
# FUTURE JUDGE TEMPLATES (UNCOMMENT TO ENABLE)
# =====================================================================
#
# class ToneJudge:
#     """Example Phoenix LLM-as-a-judge for subjective business logic.
#
#     Keep subjective checks behind an explicit LLM evaluator. Deterministic
#     formatting checks, like source-reference format, should stay regex-based.
#     """
#
#     name = "professional_tone_judge"
#
#     def __init__(self, eval_model):
#         from phoenix.evals import LLMEvaluator
#
#         prompt_template = """
#         You are evaluating the tone of an AI assistant's response.
#         The tone must be professional, technical, and objective.
#
#         Response to evaluate:
#         {output}
#
#         Answer strictly with 'Professional' or 'Unprofessional'.
#         """
#
#         self.evaluator = LLMEvaluator(
#             model=eval_model,
#             template=prompt_template,
#             output_parser=lambda resp: 1.0
#             if "Professional" in resp
#             else 0.0,
#         )
#
#     def evaluate(self, answer: str, **kwargs):
#         score = self.evaluator.evaluate({"output": answer})
#         return {"score": score}
