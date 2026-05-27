"""Reference-free LLM-as-judge evaluators via ``phoenix.evals`` v3.

Uses ``create_classifier`` with prompt templates to build
``ClassificationEvaluator`` instances for hallucination detection,
QA correctness, and relevance assessment.

These evaluators implement the ``EvalsEvaluator`` protocol, so
``client.experiments.run_experiment()`` natively recognises them.

All heavy imports are deferred into ``build_phoenix_evaluators()``.
"""

from __future__ import annotations

import logging
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

    Returns ``ClassificationEvaluator`` objects that implement the
    ``EvalsEvaluator`` protocol.  These are passed directly to
    ``client.experiments.run_experiment(evaluators=...)`` and are
    natively understood by the Phoenix experiment runner.

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
    from phoenix.evals import create_classifier

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

    # ── Build one ClassificationEvaluator per requested name ─────────
    evaluators: list = []

    for eval_name in eval_config.phoenix_evaluators:
        spec = _EVALUATOR_SPECS.get(eval_name)
        if spec is None:
            logger.warning("Unknown Phoenix evaluator '%s' — skipping", eval_name)
            continue

        prompt_template, choices, direction = spec

        evaluator = create_classifier(
            name=eval_name,
            prompt_template=prompt_template,
            llm=phoenix_llm,
            choices=choices,
            direction=direction,
        )

        evaluators.append(evaluator)
        logger.info("Registered Phoenix evaluator: %s", eval_name)

    return evaluators
