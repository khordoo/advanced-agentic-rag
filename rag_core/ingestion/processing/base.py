from abc import ABC, abstractmethod

from rag_core.schemas import ClaimedSourceAsset, ParentChunk

class BaseProcessor(ABC):
    @abstractmethod
    def process_asset(self, asset: ClaimedSourceAsset) -> list[ParentChunk]:
        """Reads the URI, converts to Markdown, extracts tables, and chunks it."""
        pass

class HierarcicalProcessor(BaseProcessor):
    def process_asset(self, asset: ClaimedSourceAsset) -> list[ParentChunk]:
        """Reads the URI, converts to Markdown, extracts tables, and chunks it."""
        print('Processing chunks')
