from abc import ABC,abstractmethod 
from typing import Any,List,Dict
from langchain_core.documents import  Document
from rag_core.config import RetrivalSettings

class BaseRetriever(ABC):
    
    #todo: not a good name , we will be doing  a hybrid search
    @abstractmethod
    def search(self,query:str,scopes:list[str] | None=None, limit: int | None = None,
        score_threshold: float | None = None,)->List[Document]:
        """Executes a semantic search with fallback logic."""
        raise NotImplementedError
    
    @abstractmethod
    def exact_lookup(self,filters:dict,scopes:list[str]=None,limit:int=None)->list[Document]:
        """Executes a deterministic metadata lookup."""
        raise NotImplementedError
        

