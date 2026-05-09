from pydantic import BaseModel, Field
from typing import List


class Citation(BaseModel):
    document_index: int = Field(
        ..., description="The index of the document in the retrieved documents list"
    )
    file_name: str = Field(..., description="The human-readable display name of the file")
    source_url: str = Field(default="", description="The source path or URL of the document")
    page_number: int = Field(..., description="The page number")


class RagFinalResponseSchema(BaseModel):
    answer: str = Field(..., description="The answer to the user's question")
    citations: List[Citation] = Field(..., description="The citations for the answer")
