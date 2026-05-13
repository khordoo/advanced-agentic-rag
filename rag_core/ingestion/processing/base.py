"""Abstract base for document processors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_core.schemas import ClaimedSourceAsset, ParentChunk


class BaseProcessor(ABC):
    """Contract for asset processing implementations.

    A processor is responsible for the full transformation pipeline:
    read source → convert to intermediate format → chunk → return ParentChunks.

    The processor outputs **text only**. Embedding is the responsibility of
    the downstream :class:`~rag_core.indexing.base.BaseIndexer`.
    """

    @abstractmethod
    def process_asset(self, asset: ClaimedSourceAsset) -> list[ParentChunk]:
        """Transform a claimed source asset into indexable chunks.

        Parameters
        ----------
        asset:
            An immutable snapshot of a claimed source asset containing
            the source URI, tags, scopes, and metadata needed for processing.

        Returns
        -------
        list[ParentChunk]
            Structured parent/child chunks ready for indexing.
        """
