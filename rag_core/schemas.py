"""Pydantic models shared across the pipeline."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class QAPair(BaseModel):
    """A single question-answer pair for evaluation."""

    question: str = Field(
        description="A complex, specific question that requires reading the content"
    )
    ground_truth: str = Field(
        description="The exact, factual answer based ONLY on the context."
    )


class SyntheticsTestSet(BaseModel):
    """A set of synthetic QA pairs generated from a document chunk."""

    items: List[QAPair]
    refusal: Optional[str] = None


class QARecord(BaseModel):
    """A single evaluation record with retrieval context."""

    user_input: str
    response: str
    retrieved_contexts: List[str]
    refusal: Optional[str] = None


class PromptConfig(BaseModel):
    """Parsed YAML prompt template."""

    name: str
    description: Optional[str] = None
    temperature: float = 0.2
    template: str


# Required for forward-reference resolution in Python 3.10+
SyntheticsTestSet.model_rebuild()
