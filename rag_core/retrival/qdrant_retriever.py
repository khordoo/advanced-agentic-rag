from rag_core.retrival.base import BaseReriever
from qdrant_client import QdrantClient

class QdrantRetriever(BaseReriever):
    def __init__(self,client) -> None:
        super().__init__()
        self.client= client 
    def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        
        return super().search(query, **kwargs)