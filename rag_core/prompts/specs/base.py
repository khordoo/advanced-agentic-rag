from pydantic import BaseModel, Field
from typing import List
import json


class Citation(BaseModel):
    document_index: int = Field(
        ..., description="The index of the document in the retrieved documents list"
    )
    file_name: str = Field(..., description="The name of the file")
    page_number: int = Field(..., description="The page number")


class RagFinalResponseSchema(BaseModel):
    answer: str = Field(..., description="The answer to the user's question")
    citations: List[Citation] = Field(..., description="The citations for the answer")
