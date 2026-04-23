from pydantic import BaseModel, Field


class RagVerifierResponse(BaseModel):
    grounded: bool = Field(description="Whether the answer is grounded in the context")
    reason: str = Field(description="Reason for the grounding judgment")


OutputSchema = RagVerifierResponse
