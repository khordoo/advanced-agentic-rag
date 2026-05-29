"""Reference-free LLM-as-judge evaluators for Phoenix experiments.

The Phoenix ``create_classifier`` helper is too brittle for the configured
judge model in this project: it can fail on missing tool calls or return
labels that do not match the allowed set. These evaluators keep Phoenix
experiment wiring intact while adding a resilient fallback path that:

- prefers Phoenix's structured/classification path when it works,
- falls back to raw-text generation when the classifier path fails, and
- coerces unexpected outputs into a conservative valid label instead of
  aborting the whole experiment.

All heavy imports are deferred into ``build_phoenix_evaluators()``.
"""

from __future__ import annotations

import logging
import json
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates (mustache)
#
# The experiment binds four keys to each evaluator call:
#   input    – the dataset example's input dict   (e.g. {"input": "…"})
#   output   – the task function's return dict     ({"response": "…",
#                                                    "retrieved_contexts": […]})
#   expected – the dataset example's output dict   (e.g. {"expected": "…"})
#   metadata – the dataset example's metadata dict
#
# Mustache's dot-access (pystache) lets us reach into nested dicts:
#   {{input.input}}              → the question string
#   {{output.response}}          → the RAG response string
#   {{output.retrieved_contexts}} → the list of retrieved context strings
#   {{expected.expected}}        → the ground-truth answer (when present)
# ---------------------------------------------------------------------------

_HALLUCINATION_PROMPT = """\
You are an expert evaluation judge. Your task is to determine whether an AI \
assistant's response is **grounded** in the retrieved context passages, or \
whether it contains **hallucinated** information that cannot be traced back \
to those passages.

### Retrieved Contexts
{{output.retrieved_contexts}}

### User Question
{{input.input}}

### AI Response
{{output.response}}

### Instructions
- A response is "factual" if every factual claim it makes is supported by \
the retrieved contexts above.
- A response is "hallucinated" if it contains any claim, fact, or detail \
that is NOT present in or inferable from the retrieved contexts.
- Minor paraphrasing or summarisation is acceptable and should be classified \
as "factual".
- If the retrieved contexts are empty and the response provides an answer, \
classify as "hallucinated".

Classify this response."""

_QA_CORRECTNESS_PROMPT = """\
You are an expert evaluation judge. Your task is to determine whether an AI \
assistant's response correctly answers the user's question.

### User Question
{{input.input}}

### AI Response
{{output.response}}

### Reference Answer
{{expected.expected}}

### Instructions
- A response is "correct" if it substantively answers the question in a way \
that is consistent with the reference answer. It does not need to be word-\
for-word identical.
- A response is "incorrect" if it fails to answer the question, provides a \
contradictory answer, or is materially wrong compared to the reference.
- If no reference answer is available, judge correctness based on the \
coherence and plausibility of the response alone.

Classify this response."""

_RELEVANCE_PROMPT = """\
You are an expert evaluation judge. Your task is to determine whether an AI \
assistant's response is **relevant** to the user's question.

### User Question
{{input.input}}

### AI Response
{{output.response}}

### Instructions
- A response is "relevant" if it directly addresses the user's question and \
provides information that helps answer it.
- A response is "irrelevant" if it discusses unrelated topics, provides \
tangential information that does not help answer the question, or is a \
refusal / "I don't know" with no useful content.

Classify this response."""


# Canonical evaluator name → (prompt, choices, direction)
_EVALUATOR_SPECS: dict[str, tuple[str, dict[str, tuple[float, str]], str]] = {
    "hallucination": (
        _HALLUCINATION_PROMPT,
        {
            "factual": (1.0, "The response is grounded in the retrieved contexts"),
            "hallucinated": (0.0, "The response contains information not found in the contexts"),
        },
        "maximize",
    ),
    "qa_correctness": (
        _QA_CORRECTNESS_PROMPT,
        {
            "correct": (1.0, "The response correctly answers the question"),
            "incorrect": (0.0, "The response does not correctly answer the question"),
        },
        "maximize",
    ),
    "relevance": (
        _RELEVANCE_PROMPT,
        {
            "relevant": (1.0, "The response is relevant to the question"),
            "irrelevant": (0.0, "The response is not relevant to the question"),
        },
        "maximize",
    ),
}


def _build_prompt_template(prompt_template: str):
    from phoenix.evals.llm.prompts import PromptTemplate

    return PromptTemplate(template=prompt_template)


def _format_retrieved_contexts(value: Any, *, max_contexts: int = 8, max_chars: int = 1200) -> str:
    """Render retrieved contexts into a bounded, readable block."""
    if not isinstance(value, list) or not value:
        return "[]"

    formatted: list[str] = []
    for index, item in enumerate(value[:max_contexts], start=1):
        text = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False, default=str)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > max_chars:
            text = text[: max_chars - 3].rstrip() + "..."
        formatted.append(f"{index}. {text}")

    if len(value) > max_contexts:
        formatted.append(f"... {len(value) - max_contexts} more context(s) omitted")

    return "\n".join(formatted)


def _extract_label_from_text(text: str, labels: list[str]) -> str | None:
    """Best-effort label extraction from a raw model response."""
    if not text:
        return None

    normalized = text.strip().lower()

    # JSON first: {"label": "..."}
    try:
        payload = json.loads(normalized)
    except Exception:
        payload = None
    if isinstance(payload, dict):
        raw_label = payload.get("label")
        if isinstance(raw_label, str):
            candidate = raw_label.strip().lower()
            if candidate in labels:
                return candidate

    # Exact word-boundary match, preferring the first label listed.
    for label in labels:
        if re.search(rf"(?<!\\w){re.escape(label.lower())}(?!\\w)", normalized):
            return label

    return None


def _conservative_fallback_label(choices: dict[str, tuple[float, str]]) -> str:
    """Return the lowest-scoring label as a safe fallback."""
    return min(choices.items(), key=lambda item: item[1][0])[0]


class _RobustClassificationEvaluator:
    """Phoenix-compatible evaluator with resilient classification fallback."""

    def __init__(
        self,
        *,
        name: str,
        llm: Any,
        prompt_template: str,
        choices: dict[str, tuple[float, str]],
        direction: str,
    ) -> None:
        self.name = name
        self.llm = llm
        self._prompt_template = _build_prompt_template(prompt_template)
        self._choices = choices
        self._labels = list(choices.keys())
        self._direction = direction
        self.direction = direction
        self._fallback_label = _conservative_fallback_label(choices)
        self.source = "llm"
        self.input_schema = None

    def _build_prompt(self, eval_input: dict[str, Any]) -> str:
        variables = dict(eval_input)
        output = variables.get("output")
        if isinstance(output, dict) and "retrieved_contexts" in output:
            output = dict(output)
            output["retrieved_contexts"] = _format_retrieved_contexts(output["retrieved_contexts"])
            variables["output"] = output

        messages = self._prompt_template.render(variables=variables)
        if isinstance(messages, list) and messages:
            first = messages[0]
            content = getattr(first, "content", None)
            if isinstance(content, str):
                return content
        return str(messages)

    def _score_from_label(self, label: str):
        score, _description = self._choices[label]
        return score

    def _raw_fallback_prompt(self, prompt: str) -> str:
        allowed_labels = ", ".join(self._labels)
        return (
            prompt
            + "\n\nReturn only a JSON object with keys 'label' and 'explanation'. "
            + f"The label must be exactly one of: {allowed_labels}."
        )

    def _coerce_response(self, response: Any, fallback_text: str) -> tuple[str | None, str | None]:
        label: str | None = None
        explanation: str | None = None

        if isinstance(response, dict):
            raw_label = response.get("label")
            if isinstance(raw_label, str):
                candidate = raw_label.strip().lower()
                if candidate in self._labels:
                    label = candidate
            raw_explanation = response.get("explanation")
            if isinstance(raw_explanation, str) and raw_explanation.strip():
                explanation = raw_explanation.strip()

        if label is None and fallback_text:
            label = _extract_label_from_text(fallback_text, self._labels)

        return label, explanation

    def evaluate(self, eval_input: dict[str, Any], input_mapping: Any | None = None):
        from phoenix.evals.evaluators import Score

        prompt = self._build_prompt(eval_input)
        explanation: str | None = None

        try:
            response = self.llm.generate_classification(
                prompt=prompt,
                labels=self._labels,
                include_explanation=True,
            )
            label, explanation = self._coerce_response(response, prompt)
            if label is None:
                raise ValueError("Invalid classification label")
        except Exception as exc:
            logger.warning(
                "Phoenix classifier path failed for %s; falling back to raw text: %s",
                self.name,
                exc,
            )
            try:
                fallback_text = self.llm.generate_text(self._raw_fallback_prompt(prompt))
            except Exception as fallback_exc:
                logger.warning(
                    "Phoenix raw-text fallback failed for %s; using conservative label: %s",
                    self.name,
                    fallback_exc,
                )
                label = self._fallback_label
            else:
                label, explanation = self._coerce_response({}, fallback_text)
                if label is None:
                    label = self._fallback_label

        score = self._score_from_label(label)
        return [
            Score(
                name=self.name,
                score=score,
                label=label,
                explanation=explanation,
                metadata={"model": getattr(self.llm, "model", None)},
                direction=self._direction,
                kind="llm",
            )
        ]

    async def async_evaluate(self, eval_input: dict[str, Any], input_mapping: Any | None = None):
        from phoenix.evals.evaluators import Score

        prompt = self._build_prompt(eval_input)
        explanation: str | None = None

        try:
            response = await self.llm.async_generate_classification(
                prompt=prompt,
                labels=self._labels,
                include_explanation=True,
            )
            label, explanation = self._coerce_response(response, prompt)
            if label is None:
                raise ValueError("Invalid classification label")
        except Exception as exc:
            logger.warning(
                "Phoenix classifier path failed for %s; falling back to raw text: %s",
                self.name,
                exc,
            )
            try:
                fallback_text = await self.llm.async_generate_text(self._raw_fallback_prompt(prompt))
            except Exception as fallback_exc:
                logger.warning(
                    "Phoenix raw-text fallback failed for %s; using conservative label: %s",
                    self.name,
                    fallback_exc,
                )
                label = self._fallback_label
            else:
                label, explanation = self._coerce_response({}, fallback_text)
                if label is None:
                    label = self._fallback_label

        score = self._score_from_label(label)
        return [
            Score(
                name=self.name,
                score=score,
                label=label,
                explanation=explanation,
                metadata={"model": getattr(self.llm, "model", None)},
                direction=self._direction,
                kind="llm",
            )
        ]


# ---------------------------------------------------------------------------
# LangChain → phoenix.evals.LLM bridge
# ---------------------------------------------------------------------------


def _wrap_langchain_as_phoenix_llm(
    langchain_model: Any,
    model_name: str = "langchain-model",
) -> Any:
    """Wrap a pre-built LangChain ``BaseChatModel`` as a ``phoenix.evals.LLM``.

    Phoenix v3's ``LLM(provider=..., model=...)`` always builds its own
    client via internal factories.  This helper constructs an ``LLM``
    instance and injects the existing LangChain model into the adapter
    layer that ``LLM`` delegates to, avoiding any duplicate client
    creation or provider-mapping logic.

    Parameters
    ----------
    langchain_model:
        A LangChain ``BaseChatModel`` instance (e.g. from ``AgentFactory``).
    model_name:
        A human-readable model identifier used for tracing metadata.
    """
    from phoenix.evals.llm.wrapper import LLM
    from phoenix.evals.llm.adapters.langchain.adapter import LangChainModelAdapter

    llm = object.__new__(LLM)
    llm.provider = "langchain"
    llm.model = model_name

    adapter = LangChainModelAdapter(langchain_model, model=model_name)

    llm._sync_client = langchain_model
    llm._async_client = langchain_model
    llm._sync_adapter = adapter
    llm._async_adapter = adapter
    llm._rate_limit_errors = []
    llm._rate_limiters = []

    return llm


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_phoenix_evaluators(
    config: dict,
    eval_config,
) -> list:
    """Build reference-free Phoenix LLM-as-judge evaluator instances.

    Returns evaluator objects that implement the Phoenix evaluator protocol.
    These are passed directly to ``client.experiments.run_experiment(
    evaluators=...)`` and are natively understood by the Phoenix experiment
    runner.

    Parameters
    ----------
    config:
        Application config dict (from ``load_config()``).
    eval_config:
        ``EvalConfig`` instance.

    Returns
    -------
    list
        One ``ClassificationEvaluator`` per name in
        ``eval_config.phoenix_evaluators``.
    """
    # ── Lazy imports ─────────────────────────────────────────────────
    from rag_core.agents import AgentFactory

    # ── Build judge LLM once, wrap it for Phoenix ────────────────────
    judge_langchain = AgentFactory.create(
        eval_config.judge_provider,
        temperature=0,
        model_name=eval_config.judge_model,
        config=config,
    )

    judge_model_name = eval_config.judge_model or "default"
    phoenix_llm = _wrap_langchain_as_phoenix_llm(
        judge_langchain,
        model_name=judge_model_name,
    )

    # ── Build one evaluator per requested name ──────────────────────
    evaluators: list = []

    for eval_name in eval_config.phoenix_evaluators:
        spec = _EVALUATOR_SPECS.get(eval_name)
        if spec is None:
            logger.warning("Unknown Phoenix evaluator '%s' — skipping", eval_name)
            continue

        prompt_template, choices, direction = spec
        evaluator = _RobustClassificationEvaluator(
            name=eval_name,
            prompt_template=prompt_template,
            llm=phoenix_llm,
            choices=choices,
            direction=direction,
        )

        evaluators.append(evaluator)
        logger.info("Registered Phoenix evaluator: %s", eval_name)

    return evaluators
