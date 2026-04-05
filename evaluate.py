import json
import logging
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import evaluate,Dataset,EvaluationDataset
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,      # Formerly answer_relevance
    ContextRecall,          # Formerly context_recall
    ContextPrecision,       # Formerly context_precision
    AnswerCorrectness       # Formerly answer_correctness
)

from ragas.run_config import RunConfig
import pandas as pd
from dotenv import dotenv_values
from agents import AgentFactory
from rag import RAGPipeline

config = dotenv_values()
logger = logging.getLogger(__name__)

def load_evaluation_qa(path)->df:
    with open(path,'r',encoding='utf-8') as f:
        data=json.load(f)
    return data

def build_evaluation_set(test_cases):
    rag=RAGPipeline(config, provider='aws')
    eval_samples=[]
    total_cases=len(test_cases)
    for idx,case in enumerate(test_cases):
        rag_out= rag.run(case['question'])
        eval_samples.append(
            {
                'user_input':case['question'],
                'reference': case['ground_truth'],
                'response':rag_out['response'],
                'retrieved_contexts':rag_out['contexts'],
            }
        )
        logger.info(f'completed test case {idx/total_cases}')
    ds=EvaluationDataset.from_list(eval_samples)
    return ds
def run_eval(dataset):
    logger.info('Running evaluation on the RAG response')
    max_request=int(config.get('LLM_MAX_PARALLEL_REQUEST',4))
    ragas_config = RunConfig(timeout=60,max_retries=3,max_workers=max_request)
    # llm_judge = AgentFactory.create('aws', temperature=0)
    llm_judge = AgentFactory.create('lmstudio', temperature=0)
    # llm_judge = AgentFactory.create('ollama', temperature=0)
    # llm_judge = AgentFactory.get_judge_model()
    ragas_vertex_llm=LangchainLLMWrapper(llm_judge)
    hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)
    metrics = [
        Faithfulness(),
        AnswerRelevancy(),      # Formerly answer_relevance
        ContextRecall(),          # Formerly context_recall
        ContextPrecision(),       # Formerly context_precision
        AnswerCorrectness(weights=[0.8, 0.2]) # [semantic_weight, factual_overlap_weight]))   # Give more weight to Semantic Similarity (meaning) # and less weight to Factual Overlap (exact words)
    ]
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_vertex_llm,
        embeddings=hf_embeddings,
        run_config=ragas_config
    )
    return result

def run_eval_pipeline():
     test_cases=load_evaluation_qa(config['SYNTHETIC_QA_PERSIST_PATH'])
     eval_dataset= build_evaluation_set(test_cases)
     result = run_eval(eval_dataset)
     print(result)
     df_result = result.to_pandas()
     df_result.to_csv('data/synthetic/eval_results_detailed.csv')
    # df = pd.DataFrame(data)

    # print('df:',df.head(

    # ))
    # dataset=EvaluationDataset.from_pandas(df,
    # # name='synthetic_eval_set',
    # # backend='inmemory'
    # )
    # return dataset

# ds=load_regas_dataset(config['SYNTHETIC_QA_PERSIST_PATH'])


# print(result)
# df_result = result.to_pandas()
# df_result.to_csv('eval_results_detailed.csv')
if __name__ =='__main__':
    run_eval_pipeline()
