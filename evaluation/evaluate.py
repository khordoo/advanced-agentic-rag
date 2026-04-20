"""RAGAS-based evaluation pipeline for the Advanced RAG system."""

from __future__ import annotations
import re
import json
import logging

import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate, EvaluationDataset
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerCorrectness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)
from ragas.run_config import RunConfig

from rag_core.agents import AgentFactory
from rag_core.config import load_config
from rag_core.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


def load_evaluation_qa(path: str) -> list[dict]:
    """Load QA test cases from a JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_evaluation_set(
    test_cases: list[dict],
    config: dict,
    provider: str = "aws",
    prompt_name: str = "rag_eval",
) -> EvaluationDataset:
    """Run RAG on each test case and build a RAGAS evaluation dataset."""
    rag = RAGPipeline(config, provider=provider, prompt_name=prompt_name)
    samples = []

    for idx, case in enumerate(test_cases):
        rag_out = rag.run(case["question"])
        # remvoe file name references from the response, RAGAS penalizes for this
        rag_out["response"] = re.sub(r"\s*\[\[.*?\]\]", "", rag_out["response"])
        samples.append(
            {
                "user_input": case["question"],
                "reference": case["ground_truth"],
                "response": rag_out["response"],
                "retrieved_contexts": [c.page_content for c in rag_out["contexts"]],
            }
        )
        logger.info("Completed test case %d/%d", idx + 1, len(test_cases))

    return EvaluationDataset.from_list(samples)


def run_eval(
    dataset: EvaluationDataset,
    config: dict,
    judge_provider: str = "aws",
):
    """Run RAGAS metrics on the evaluation dataset."""
    # model_name = "zai.glm-5"
    max_requests = int(config.get("LLM_MAX_PARALLEL_REQUEST", 4))
    ragas_config = RunConfig(timeout=60, max_retries=3, max_workers=max_requests)

    llm_judge = AgentFactory.create(judge_provider, temperature=0, config=config)
    ragas_llm = LangchainLLMWrapper(llm_judge)

    hf_emb = HuggingFaceEmbeddings(
        model_name=config.get(
            "HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    ragas_emb = LangchainEmbeddingsWrapper(hf_emb)

    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextRecall(),
        ContextPrecision(),
        AnswerCorrectness(weights=[0.8, 0.2]),
    ]

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=hf_emb,
        run_config=ragas_config,
    )
    return result


def run_eval_pipeline(
    config: dict | None = None, provider: str = "aws", prompt_name: str = "rag_eval"
):
    """End-to-end: load QA set → run RAG → evaluate → save results."""
    config = config or load_config()

    test_cases = load_evaluation_qa(config["SYNTHETIC_QA_PERSIST_PATH"])
    eval_dataset = build_evaluation_set(
        test_cases, config, provider=provider, prompt_name=prompt_name
    )
    result = run_eval(eval_dataset, config)

    print(result)

    df_result = result.to_pandas()
    output_path = "data/synthetic/eval_results_detailed.csv"
    df_result.to_csv(output_path, index=False)
    logger.info("Results saved to %s", output_path)

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_eval_pipeline()
