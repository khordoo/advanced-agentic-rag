from langchain_core.documents import Document
from pydantic import BaseModel, Field
from pydantic.v1.errors import NoneIsAllowedError
from rag_core.retrival.base import BaseRetriever
from rag_core.retrival.qdrant_retriever import QdrantRetriever
from rag_core.schemas import VectorDatabasePayloadKeys as PayloadSchema
from langchain_core.tools import StructuredTool


class Query(BaseModel):
    query: str = Field(description="The query to be answered")


class SeachToolSchema(BaseModel):
    query: str = Field(description="The query to be answered")


class TableLookupToolSchema(BaseModel):
    table_id: str = Field(description="exact id of the table for search")
    file_name: str = Field(description="name of the file that table exist in")


class FetchPageToolSchema(BaseModel):
    file_name: str = Field(description="name of the file")
    page_number: int = Field(description="exact page number to get")


class AgentToolFactory:
    """
    Factory class for creating tools for the  langgraph agent.
    """

    def __init__(self, retriever: BaseRetriever | None = None) -> None:
        self.retriever = retriever or QdrantRetriever()
        self._tools: list[StructuredTool] | None = None

    def _semantic_search(self, query: str) -> list[Document]:
        docs = self.retriever.search(query=query)
        return docs

    def _page_lookup(self, file_name: str, page_number: int):
        filters = {
            PayloadSchema.FILE_NAME.value: file_name,
            PayloadSchema.PAGE.value: [page_number],
        }
        return self.retriever.exact_lookup(filters=filters)

    def _table_lookup(self, table_id: str, file_name: str):
        filters = {
            PayloadSchema.FILE_NAME.value: file_name,
            PayloadSchema.TABLE_ID.value: table_id,
        }
        return self.retriever.exact_lookup(filters=filters)

    @classmethod
    def create_tools(cls, retriever: BaseRetriever | None=None) -> list[StructuredTool]:
        instance = cls(retriever or QdrantRetriever())
        return instance._get_tools()

    def _get_tools(self) -> list[StructuredTool]:
        if self._tools is None:
            self._tools = [
                StructuredTool.from_function(
                    func=self._semantic_search,
                    name="semantic_search",
                    description="Semantic search tool to find relevant documents",
                    args_schema=SeachToolSchema,
                ),
                StructuredTool.from_function(
                    func=self._page_lookup,
                    name="page_lookup",
                    description="Page lookup tool to find documents on a specific page",
                    args_schema=FetchPageToolSchema,
                ),
                StructuredTool.from_function(
                    func=self._table_lookup,
                    name="table_lookup",
                    description="Table lookup tool to find documents in a specific table",
                    args_schema=TableLookupToolSchema,
                ),
            ]
        return self._tools

if __name__=="__main__":
    tools= AgentToolFactory.create_tools()
    print(tools)
