from enum import StrEnum
from pydantic import BaseModel


class IntentCategory(StrEnum):
    DOMAIN_EXPERT = "domain_expert"
    GENERAL_ASSISTANT = "general_assistant"

    @classmethod
    def get_categories(cls):
        return [category.value for category in cls]


class IntentRouterResponse(BaseModel):
    reasoning: str
    intent: IntentCategory


class IntentExamples(BaseModel):
    message: str
    response: IntentRouterResponse


EXAMPLES = [
    IntentExamples(
        message="Hi there!",
        response=IntentRouterResponse(
            reasoning="The user is greeting me. This is casual conversation.",
            intent=IntentCategory.GENERAL_ASSISTANT,
        ),
    ),
    IntentExamples(
        message="What was the revenue for Gillette in Q3?",
        response=IntentRouterResponse(
            reasoning="The user is asking for specific financial data. I need to route this to the research tools.",
            intent=IntentCategory.DOMAIN_EXPERT,
        ),
    ),
    IntentExamples(
        message="Who are you and what is your name?",
        response=IntentRouterResponse(
            reasoning="The user is asking about my identity. This does not require financial documents.",
            intent=IntentCategory.GENERAL_ASSISTANT,
        ),
    ),
]
