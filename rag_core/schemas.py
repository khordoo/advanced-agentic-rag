"""Pydantic models shared across the pipeline."""

from __future__ import annotations
from enum import Enum
from typing import List, Optional,Any
from langchain_core.documents import Document
from pydantic import BaseModel, Field, ConfigDict
from dataclasses import dataclass, field
from qdrant_client import  models

class QAPair(BaseModel):
    """A single question-answer pair for evaluation."""

    question: str = Field(
        description="A complex, specific question that requires reading the content"
    )
    ground_truth: str = Field(
        description="The exact, factual answer based ONLY on the context."
    )


class SyntheticsTestSet(BaseModel):
    """A set of synthetic QA pairs generated from a document chunk."""

    items: List[QAPair]
    refusal: Optional[str] = None


class QARecord(BaseModel):
    """A single evaluation record with retrieval context."""

    user_input: str
    response: str
    retrieved_contexts: List[str]
    refusal: Optional[str] = None




class PromptConfig(BaseModel):
    """Parsed YAML prompt template."""

    name: str
    description: Optional[str] = None
    temperature: float = 0.2
    template: str

#------------- Ingestion -------------------
# --- 1. The Pydantic Schema (The Contract) ---
class TableSummary(BaseModel):
    """Schema for the LLM to geenrate a sumary of the table"""
    table_title: str = Field(description="The exact name or number of the table (e.g., 'Table 4.1'). If none, write 'Unnamed Table'.")
    retrieval_summary: str = Field(description="A concise 2-sentence summary of the core trends, anomalies, or totals in the table.")
    column_headers: list[str] = Field(description="Extract every column header across the top of the table.")
    row_headers: list[str] = Field(description="Extract the main row categories down the left side of the table.")
    keywords: list[str] = Field(description="5-7 highly specific entities, financial metrics, or jargon terms found in the data.")
    question_signals: list[str] = Field(description="3 highly specific questions a user might ask which this table perfectly answers. Frame them exactly how a human would type a search query.")
    fine_print_and_qualifiers: str = Field(
            description="Extract any footnotes, captions, or fine print from the surrounding context that change how the numbers should be interpreted (e.g., 'Unaudited', 'Dollars in thousands', 'Excludes Q4'). If none exist, write 'None'."
        )

    def to_markdown(self) -> str:
            """Converts the JSON state into an atomic, Agent-friendly Markdown block."""

            # 1. We use a single H3 for the main title to group it under the document's H2s
            md = f"### Data Table: {self.table_title}\n\n"

            # 2. We use bold text for keys to prevent accidental chunking
            md += f"**Summary:** {self.retrieval_summary}\n\n"

            # 3. We join lists with commas for token-dense readability
            md += f"**Columns:** {', '.join(self.column_headers)}\n"
            md += f"**Rows:** {', '.join(self.row_headers)}\n"
            md += f"**Keywords:** {', '.join(self.keywords)}\n\n"

            # 4. We use bullet points for questions
            md += "**Can Answer:**\n"
            for q in self.question_signals:
                md += f"- {q}\n"

            # 5. We use a blockquote for the fine print so the LLM knows it's a disclaimer
            if self.fine_print_and_qualifiers and self.fine_print_and_qualifiers.lower() != "none":
                md += f"\n> **Fine Print/Qualifiers:** {self.fine_print_and_qualifiers}\n"

            return md

class TableChunk(BaseModel):
    """ Used to store in databse"""
    table_id: str
    parent_id:str
    summary: TableSummary

class ExtractedTable(BaseModel):
    # 1. The Atomic Parts (Stored Separately)
    raw_markdown_table: str
    context_above: str
    context_below: str

    # 2. The LLM Output (Added later in the pipeline)
    seo_metadata: TableMetadata | None = None

    @property
    def extraction_context(self)->str:
        return f'{self.context_above}\n\n{self.raw_markdown_table}\n\n{self.context_below}'


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ParentChunk:
    """A parent chunk with its child documents ready for storage."""

    id: str
    enriched_text: str
    metadata: ParentMetadata
    children: list[Document] = field(default_factory=list)


class BaseChunkMetadata(BaseModel):
    """
    Base schema for document chunk metadata.

    Attributes:
        file_name: The original name of the source file.
        source: The URI or path source of the document.
        pages: List of page numbers associated with this chunk [1,2] or [1].
        tags: Categorization labels used for logical grouping (e.g., ['finance', 'legal']).
        scopes: Access control scopes used for RBAC filtering (defaults to ['public']).
    """
    model_config = ConfigDict(extra='allow')

    file_name: str
    source: str
    pages: list[int]
    tags: list[str] = Field(default_factory=list)
    scopes: list[str] = Field(default_factory=lambda: ["public"])


# 2. The Parent (Adds Table context if needed)
class ParentMetadata(BaseChunkMetadata):
    table_id: Optional[str] = None


# 3. The Child (Adds the mandatory link back to the Parent)
class ChildMetadata(BaseChunkMetadata):
    parent_id: str


class VectorDatabasePayloadKeys(str,Enum):
    SOURCE = 'source'
    FILE_NAME = 'file_name'
    PAGE='pages'
    TABLE_ID='table_id'
    TEXT='text'
    PARENT_ID='parent_id'
    SCOPES='scopes'
    TAGS='tags'

_BASE_INDEX_SCHEMA={
    VectorDatabasePayloadKeys.SOURCE.value: models.PayloadSchemaType.KEYWORD,
    VectorDatabasePayloadKeys.FILE_NAME.value: models.PayloadSchemaType.KEYWORD,
    VectorDatabasePayloadKeys.PAGE.value: models.PayloadSchemaType.INTEGER,
    VectorDatabasePayloadKeys.SCOPES.value: models.PayloadSchemaType.KEYWORD,
    VectorDatabasePayloadKeys.TAGS.value: models.PayloadSchemaType.KEYWORD,
}

PARENT_INDEX_SCHEMA={
    **_BASE_INDEX_SCHEMA,
    VectorDatabasePayloadKeys.TABLE_ID.value: models.PayloadSchemaType.KEYWORD,
}
CHILD_INDEX_SCHEMA={
    **_BASE_INDEX_SCHEMA,
    VectorDatabasePayloadKeys.PARENT_ID.value: models.PayloadSchemaType.KEYWORD,
}

# Required for forward-reference resolution in Python 3.10+
SyntheticsTestSet.model_rebuild()
