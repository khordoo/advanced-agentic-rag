import pickle
import json
import logging
from dotenv import dotenv_values
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from langchain_chroma import Chroma
from agents import AgentFactory
from agents import PromptLoader

config=dotenv_values()
logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self,config:dict):
        self.config=config
        try:
            with open(config['DOCUMENT_STORE_PERSIST_PATH'],'rb') as f:
                self.store=pickle.load(f)
            logger.info('Succefully loaded the document store.')
        except FileNotFoundError:
            logger.error(f"Doc store not found at {config['DOCUMENT_STORE_PERSIST_PATH']}")
            self.store={}

        emb = HuggingFaceEmbeddings(model_name=config['HF_EMBEDDING_MODEL'])
        self.vector_db = Chroma(
               collection_name=config['CHROMA_COLLECTION_NAME'],
               embedding_function=emb,
               persist_directory=config['CHROMA_PERSIST_DIRECTORY']
           )
        self.reranker =CrossEncoder(config.get('HF_RERANKER_MODEL'))
    def similarity_search(self,query,k):
       return self.vector_db.similarity_search(query,k=k)

    def get_document(self,id):
        return self.store.get(id)

    def _rerank(self,query,docs,top_n=3):
        top_n=int(self.config.get('RERANK_TOP_N',3))
        docs_fused= [(query,doc.page_content) for doc in docs]
        scores = self.reranker.predict(docs_fused)
        print('Raw rerank score:',scores)
        scores_docs = list(zip(scores,docs))
        logger.info('Reranker scores:',scores_docs)
        print('Rerank score:',scores_docs)
        scores_docs.sort(key=lambda x:x[0], reverse=True)
        # print('------------------sorted---------------')
        # for s in scores_docs:
        #     print(s)
        #     print('----------------s------------------------------------')
        print('Soreted Rerank score soreted:',scores_docs)
        ranked_docs= [doc for _,doc in scores_docs][:top_n]
        print('-----------docs after rerank---------------')
        print(ranked_docs)
        return ranked_docs

    def hierarchical_search(self,query,k=10,top_n=3):
        short_docs = self.similarity_search(query,k=15)
        print('short docs:--------- before rerank')
        short_docs = self._rerank(query,short_docs)
        seen_parents=set()
        parent_docs=[]
        for doc in short_docs:
           parent_id = doc.metadata.get('parent')
           if parent_id and parent_id not in seen_parents:
                seen_parents.add(parent_id)
                parent_doc=self.get_document(parent_id)
                
                if parent_doc:
                    parent_docs.append(parent_doc.get('content'))
                else:
                     logger.warning(f"Parent ID {parent_id} found in VectorDB but missing in Pickle Store.")

        logger.info(f'Deduplicated parents: {len(short_docs)} -> {len(parent_docs)}')
        # parent_docs = self._rerank(query,parent_docs)
        return parent_docs

class RAGPipeline:
    def __init__(self,config:dict, provider:str):
        self.config=config
        self.llm = AgentFactory.create(provider)
        self.retriever=Retriever(config)
        logger.info('RAG pipleine initialzed successfuly')
    def run(self,query):
        context_doc = self.retriever.hierarchical_search(query,k=10)
        context_text = '\n\n----\n\n'.join([doc for doc in context_doc])
        prompt_template = PromptLoader.load(name='rag_qa',context=context_text, question=query)
        # print('FULL prompt:-------------')
        # print(prompt_template.template)
        response=self.llm.invoke(prompt_template.template)
        return {'response': response.content,
         'contexts':context_doc
        }


if __name__ =='__main__':
    # r = Retriever()

    query="What are the key differentiators mentioned in the company's integrated strategy?"
    rag=RAGPipeline(config, provider='aws')
    res=rag.run(query)
    print('--------------Response----------------')
    print(res)
   # wihtout reranker:
   # {'faithfulness': 0.8889, 'answer_relevancy': 0.8999, 'context_recall': 0.9167, 'context_precision': 0.8611, 'answer_correctness': 0.8512}
   # {'faithfulness': 0.8333, 'answer_relevancy': 0.9010, 'context_recall': 0.9167, 'context_precision': 0.8472, 'answer_correctness': 0.8089}
   # {'faithfulness': 0.8750, 'answer_relevancy': 0.8139, 'context_recall': 0.9167, 'context_precision': 0.8750, 'answer_correctness': 0.7909}
    #with reranker
    #{'faithfulness': 0.9167, 'answer_relevancy': 0.8866, 'context_recall': 0.9167, 'context_precision': 0.8917, 'answer_correctness': 0.8428}
    #{'faithfulness': 0.8333, 'answer_relevancy': 0.8911, 'context_recall': 0.9167, 'context_precision': 0.8806, 'answer_correctness': 0.8405}
    #{'faithfulness': 0.9167, 'answer_relevancy': 0.8862, 'context_recall': 0.9167, 'context_precision': 0.8806, 'answer_correctness': 0.8365}
    # rerank after small chunks
    #{'faithfulness': 0.9722, 'answer_relevancy': 0.8977, 'context_recall': 0.9167, 'context_precision': 0.9167, 'answer_correctness': 0.8486}
    #{'faithfulness': 1.0000, 'answer_relevancy': 0.8841, 'context_recall': 0.9167, 'context_precision': 0.9167, 'answer_correctness': 0.8248}
    #{'faithfulness': 0.9722, 'answer_relevancy': 0.9029, 'context_recall': 0.9167, 'context_precision': 0.9167, 'answer_correctness': 0.8363}
    # res= r.similarity_search(query,k=3)
    # print(res)
    # print('=='*20)
    # res= r.hierarchical_search(query,k=3)
    # for r in res:
    #     print('---'*20)
    #     print(r)
