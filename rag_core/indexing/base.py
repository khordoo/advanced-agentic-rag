"""Abstract base for document indexers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_core.schemas import ClaimedSourceAsset, ParentChunk


class BaseIndexer(ABC):
    """Contract for indexer implementations (Qdrant, Pinecone, etc.)."""

    def index_asset(
        self,
        asset: ClaimedSourceAsset,
        document_blocks: list[ParentChunk],
    ) -> None:
        """Index the latest processed content for a single source asset.

        Stateless backends keep their existing behavior by forwarding to the
        legacy chunk-only ingestion method.
        """
        self.ingest_documents(document_blocks)

    @abstractmethod
    def ingest_documents(self, document_blocks: list[ParentChunk]) -> None:
        """Persist parent + child chunks into the vector store."""

    @abstractmethod
    def reset_collections(self) -> None:
        """Drop and recreate all managed collections."""
