from __future__ import annotations

from evaluation.evaluators.phoenix_judges import _EVALUATOR_SPECS
from evaluation.evaluators.phoenix_judges import _RobustClassificationEvaluator
from phoenix.client.resources.experiments.types import EvalsEvaluator


class _FakeLLM:
    def __init__(self, classify_result=None, text_result="", classify_exc=None, text_exc=None):
        self.classify_result = classify_result
        self.text_result = text_result
        self.classify_exc = classify_exc
        self.text_exc = text_exc
        self.model = "fake-judge"
        self.seen_prompts: list[str] = []

    def generate_classification(self, prompt, labels, include_explanation=True, **kwargs):
        self.seen_prompts.append(prompt)
        if self.classify_exc is not None:
            raise self.classify_exc
        return self.classify_result

    def generate_text(self, prompt, **kwargs):
        self.seen_prompts.append(prompt)
        if self.text_exc is not None:
            raise self.text_exc
        return self.text_result

    async def async_generate_classification(self, prompt, labels, include_explanation=True, **kwargs):
        return self.generate_classification(prompt, labels, include_explanation=include_explanation, **kwargs)

    async def async_generate_text(self, prompt, **kwargs):
        return self.generate_text(prompt, **kwargs)


def _example_input(long_context: str = "context") -> dict:
    return {
        "input": {"input": "What happened?"},
        "output": {
            "response": "Answer",
            "retrieved_contexts": [long_context],
        },
        "expected": {"expected": "Ref"},
        "metadata": {},
    }


def test_hallucination_falls_back_when_tool_calls_missing():
    prompt, choices, direction = _EVALUATOR_SPECS["hallucination"]
    llm = _FakeLLM(
        classify_exc=ValueError("No tool calls found in response"),
        text_result='{"label": "hallucinated", "explanation": "not grounded"}',
    )
    evaluator = _RobustClassificationEvaluator(
        name="hallucination",
        llm=llm,
        prompt_template=prompt,
        choices=choices,
        direction=direction,
    )

    result = evaluator.evaluate(_example_input())

    assert result[0].label == "hallucinated"
    assert result[0].score == 0.0
    assert "retrieved_contexts" not in llm.seen_prompts[0] or "1." in llm.seen_prompts[0]


def test_qa_correctness_recovers_from_invalid_label():
    prompt, choices, direction = _EVALUATOR_SPECS["qa_correctness"]
    llm = _FakeLLM(
        classify_result={"label": "any", "explanation": "uncertain"},
        text_result='{"label": "correct", "explanation": "matches the reference"}',
    )
    evaluator = _RobustClassificationEvaluator(
        name="qa_correctness",
        llm=llm,
        prompt_template=prompt,
        choices=choices,
        direction=direction,
    )

    result = evaluator.evaluate(_example_input())

    assert result[0].label == "correct"
    assert result[0].score == 1.0


def test_hallucination_prompt_truncates_long_contexts():
    prompt, choices, direction = _EVALUATOR_SPECS["hallucination"]
    llm = _FakeLLM(
        classify_result={"label": "factual", "explanation": "grounded"},
    )
    evaluator = _RobustClassificationEvaluator(
        name="hallucination",
        llm=llm,
        prompt_template=prompt,
        choices=choices,
        direction=direction,
    )

    long_context = "x" * 4000
    evaluator.evaluate(_example_input(long_context=long_context))

    rendered_prompt = llm.seen_prompts[0]
    assert len(rendered_prompt) < len(prompt) + 2500
    assert "..." in rendered_prompt


def test_robust_evaluator_satisfies_phoenix_protocol():
    prompt, choices, direction = _EVALUATOR_SPECS["relevance"]
    llm = _FakeLLM(classify_result={"label": "relevant", "explanation": "ok"})
    evaluator = _RobustClassificationEvaluator(
        name="relevance",
        llm=llm,
        prompt_template=prompt,
        choices=choices,
        direction=direction,
    )

    assert isinstance(evaluator, EvalsEvaluator)
