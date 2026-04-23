from enum import StrEnum

from pydantic import BaseModel


class IntentCategory(StrEnum):
    DOMAIN_EXPERT = "domain_expert"
    GENERAL_ASSISTANT = "general_assistant"

    @classmethod
    def get_categories(cls) -> list[str]:
        return [category.value for category in cls]


class IntentRouterResponse(BaseModel):
    reasoning: str
    intent: IntentCategory


OutputSchema = IntentRouterResponse
