"""Abstract base for document indexers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_core.schemas import ParentChunk


class BaseIndexer(ABC):
    """Contract for indexer implementations (Qdrant, Pinecone, etc.)."""

    @abstractmethod
    def ingest_documents(self, document_blocks: list[ParentChunk]) -> None:
        """Persist parent + child chunks into the vector store."""

    @abstractmethod
    def reset_collections(self) -> None:
        """Drop and recreate all managed collections."""