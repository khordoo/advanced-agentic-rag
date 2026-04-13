from abc import ABC,abstractmethod 
from typing import Any,List,Dict

class BaseReriever(ABC):
    
    #todo: not a good name , we will be doing  a hybrid search
    @abstractmethod
    def search(self,query:str,**kwargs)->List[Dict[str,Any]]:
        """Executes a semantic search with fallback logic."""
        pass
    
    @abstractmethod
    def exact_document_lookup(self,**kwargs)->str:
        """Executes a deterministic metadata lookup."""
        pass
        

