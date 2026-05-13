import json
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from rag_core.retrieval.base import BaseRetriever
from rag_core.retrieval.search_service import RAGSearchService
from rag_core.schemas import VectorDatabasePayloadKeys as PayloadSchema
from langchain_core.tools import StructuredTool


class Query(BaseModel):
    query: str = Field(description="The query to be answered")


class SearchToolSchema(BaseModel):
    query: str = Field(description="The query to be answered")
    asset_key: str | None = Field(
        default=None,
        description="optional exact asset_key from a previous [Asset: ...] label",
    )


class TableLookupToolSchema(BaseModel):
    table_id: str = Field(description="exact id of the table for search")
    asset_key: str = Field(description="exact asset_key that the table exists in")


class FetchPageToolSchema(BaseModel):
    asset_key: str = Field(description="exact asset_key of the source asset")
    page_number: int = Field(description="exact page number to get")


class AgentToolFactory:
    """
    Factory class for creating tools for the  langgraph agent.
    """

    def __init__(
        self,
        retriever: BaseRetriever | None = None,
        search_service: RAGSearchService | None = None,
    ) -> None:
        self.search_service = search_service or RAGSearchService(retriever=retriever)
        self.retriever = self.search_service.retriever
        self._tools: list[StructuredTool] | None = None

    def _semantic_search(self, query: str, asset_key: str | None = None) -> str:
        docs = self.search_service.search(query=query, asset_key=asset_key)
        return self._format_docs(docs)

    def _page_lookup(self, asset_key: str, page_number: int) -> str:
        filters = {
            PayloadSchema.PAGE.value: [page_number],
        }
        docs = self.retriever.exact_lookup(asset_key=asset_key, filters=filters)
        return self._format_docs(docs)

    def _table_lookup(self, table_id: str, asset_key: str) -> str:
        filters = {
            PayloadSchema.TABLE_ID.value: table_id,
        }
        docs = self.retriever.exact_lookup(asset_key=asset_key, filters=filters)
        return self._format_docs(docs)

    def _format_docs(self, docs: list[Document]) -> str:
        doc_dicts = [
            {"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs
        ]
        return json.dumps(doc_dicts)

    @classmethod
    def create_tools(
        cls,
        retriever: BaseRetriever | None = None,
        search_service: RAGSearchService | None = None,
    ) -> list[StructuredTool]:
        instance = cls(retriever=retriever, search_service=search_service)
        return instance._get_tools()

    def _get_tools(self) -> list[StructuredTool]:
        if self._tools is None:
            self._tools = [
                StructuredTool.from_function(
                    func=self._semantic_search,
                    name="semantic_search",
                    description="Semantic search tool to find relevant documents",
                    args_schema=SearchToolSchema,
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
