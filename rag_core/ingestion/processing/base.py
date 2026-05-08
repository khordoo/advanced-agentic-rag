from abc import ABC, abstractmethod
from typing import List
from rag_core.schemas import ParentChunk
from rag_core.store.postgres.models.ingestion import SourceAsset

class BaseProcessor(ABC):
    @abstractmethod
    def process_asset(self, asset: SourceAsset) -> List[ParentChunk]:
        """Reads the URI, converts to Markdown, extracts tables, and chunks it."""
        pass

class HierarcicalProcessor(BaseProcessor):
    def process_asset(self, asset: SourceAsset) -> List[ParentChunk]:
        """Reads the URI, converts to Markdown, extracts tables, and chunks it."""
        print('Processing chunks')
