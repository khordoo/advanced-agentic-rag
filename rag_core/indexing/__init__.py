"""Document indexing: vector store backends and factory."""

from rag_core.indexing.base import BaseIndexer
from rag_core.indexing.factory import IndexerBackend, create_indexer

__all__ = [
    "BaseIndexer",
    "IndexerBackend",
    "create_indexer",
]
