from typing import TypedDict, Annotated
from langgraph.graph import add_messages
import operator


from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


class RagState(TypedDict):
    # query:str
    messages: Annotated[list[BaseMessage], add_messages]
    retrieved_context: list[Document]
    source_docs: list[Document]
    token_budget: int
    iteration_count: int
    max_iterations: int
    route: str
    is_grounded: bool
    revision_attempts: int
