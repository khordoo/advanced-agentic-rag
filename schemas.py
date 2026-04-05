
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional,List

class QAPair(BaseModel):
    question: str = Field(description='A complex, specific question that requires reading the content')
    ground_truth:str = Field(description='The exact,factual answer based ONLY on the contenxt.')

class SyntheticsTestSet(BaseModel):
    items: List[QAPair]
    refusal:Optional[str]=None

class QARecord(BaseModel):
    user_input: str
    response: str
    retrieved_contexts: List[str]
    refusal: Optional[str] =None
    

class PromptConfig(BaseModel):
    name: str
    description: Optional[str]= None
    temperature: float =0.2
    template: str

#Fix his error when using Python 3.10+ type hinting 
SyntheticsTestSet.model_rebuild()