from rag_core.prompts.specs.base import RagFinalResponseSchema

EXAMPLES = [
    RagFinalResponseSchema(
        answer="Hello! I am your financial analyst assistant. What financial data or reports can I look up for you today?",
        citations=[],
    )
]
