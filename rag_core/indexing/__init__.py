"""Document indexing: vector store backends and factory."""

from rag_core.indexing.base import BaseIndexer
from rag_core.indexing.factory import (
    IndexingCompatibilityResult,
    IndexerBackend,
    create_indexer,
    validate_indexing_runtime_compatibility,
)

__all__ = [
    "BaseIndexer",
    "IndexingCompatibilityResult",
    "IndexerBackend",
    "create_indexer",
    "validate_indexing_runtime_compatibility",
]
