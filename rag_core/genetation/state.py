from typing import NotRequired
from typing import TypedDict, Annotated
from langgraph.graph import add_messages
from pydantic import BaseModel

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from rag_core.prompts.packages.rag_verifier.schema import RagVerifierResponse


class RagState(TypedDict):
    # query:str
    messages: Annotated[list[BaseMessage], add_messages]
    source_docs: list[Document]
    token_budget: int
    iteration_count: NotRequired[int]
    tool_call_count: NotRequired[int]
    intent_route: str
    verifier_response: NotRequired[RagVerifierResponse]
    self_correction_attempts: NotRequired[int]
