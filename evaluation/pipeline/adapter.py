"""Pure conversion functions between Phoenix, RAG output, and RAGAS shapes.

No heavy dependencies — stdlib ``re`` only.
"""

from __future__ import annotations

import re

# Regex that strips inline file-reference markers such as [[filename.pdf]].
# Preserves the legacy pattern from evaluation/evaluate.py.
_FILE_REF_RE = re.compile(r"\s*\[\[.*?\]\]")


def get_example_value(example, key: str, default=""):
    """Read a value from Phoenix v5 dict rows or v16 DatasetExample objects."""
    if isinstance(example, dict):
        value = example.get(key, default)
    else:
        attr_key = "output" if key == "expected" else key
        value = getattr(example, attr_key, default)

    if isinstance(value, dict):
        return value.get(key, value.get("input", value.get("expected", default)))
    return value if value is not None else default


def strip_file_refs(text: str) -> str:
    """Remove ``[[file_ref]]`` markers from RAG responses.

    RAGAS penalises extraneous tokens that don't match the ground truth,
    so these markers must be stripped before evaluation.
    """
    return _FILE_REF_RE.sub("", text)


def to_ragas_sample(example: dict, output: dict) -> dict:
    """Convert a Phoenix example + RAG output into a RAGAS sample dict.

    Parameters
    ----------
    example:
        Row from the Phoenix Dataset.  Expected keys: ``input``,
        ``expected`` (optional).
    output:
        Return value from :func:`evaluation.pipeline.task.run_rag_for_example`.
        Expected keys: ``response``, ``retrieved_contexts``.

    Returns
    -------
    dict
        ``{user_input, reference, response, retrieved_contexts}`` ready for
        ``EvaluationDataset.from_list([sample])``.
    """
    return {
        "user_input": get_example_value(example, "input", ""),
        "reference": get_example_value(example, "expected", ""),
        "response": strip_file_refs(output.get("response", "")),
        "retrieved_contexts": output.get("retrieved_contexts", []),
    }
