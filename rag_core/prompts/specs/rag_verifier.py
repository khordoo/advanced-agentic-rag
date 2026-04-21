from pydantic import BaseModel, Field


class RagVerifierResponse(BaseModel):
    grounded: bool = Field(description="Whether the answer is grounded in the context")
    reason: str = Field(description="Reason for the answer being grounded or not")


EXAMPLES = [
    RagVerifierResponse(grounded=True, reason="The answer is grounded in the context."),
    RagVerifierResponse(
        grounded=False,
        reason="The answer claims revenue grew 12%, but that figure is not found in the documents.",
    ),
]
