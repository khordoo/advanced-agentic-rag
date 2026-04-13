from abc import ABC, abstractmethod
from qdrant_client import QdrantClient
from typing import List,Dict,Any, Iterable

class BaseIndexer(ABC):

    @abstractmethod
    def ingest_documents(self,chunks: list[str], metadata: list[dict]):
        pass

    @abstractmethod
    def reset_collections(self):
        pass