from pydantic import BaseModel

from rag_core.prompts.packages.intent_router.schema import (
    IntentCategory,
    IntentRouterResponse,
)


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
    IntentExamples(
        message="USER: Good morning!",
        response=IntentRouterResponse(
            reasoning="The final message is a simple greeting. This is casual conversation.",
            intent=IntentCategory.GENERAL_ASSISTANT,
        ),
    ),
    IntentExamples(
        message="USER: What were the main factors behind the change in net sales for Beauty?",
        response=IntentRouterResponse(
            reasoning="The user is asking a specific financial question requiring document retrieval.",
            intent=IntentCategory.DOMAIN_EXPERT,
        ),
    ),
    IntentExamples(
        message="USER: What was the gross margin?\nAI: The gross margin was 51.4%.\nUSER: Are you sure that's right? Where did you find that?",
        response=IntentRouterResponse(
            reasoning="The final message is a follow-up asking for verification of a technical metric. It requires the domain expert to check the sources.",
            intent=IntentCategory.DOMAIN_EXPERT,
        ),
    ),
    IntentExamples(
        message="USER: Can you summarize the Gillette impairment?\nAI: P&G recorded a $1.3 billion impairment charge.\nUSER: Why did it drop so much?",
        response=IntentRouterResponse(
            reasoning="The pronoun 'it' in the final message refers to the financial impairment charge. This is a domain-specific follow-up.",
            intent=IntentCategory.DOMAIN_EXPERT,
        ),
    ),
    IntentExamples(
        message="USER: Explain how the evaluation framework works.\nAI: It uses NDCG and re-ranking metrics.\nUSER: Got it, thanks! How is your day going?",
        response=IntentRouterResponse(
            reasoning="Despite the technical history, the final user message is a hard pivot to purely conversational chit-chat.",
            intent=IntentCategory.GENERAL_ASSISTANT,
        ),
    ),
]
