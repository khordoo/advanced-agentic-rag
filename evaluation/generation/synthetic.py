"""Async synthetic QA generation → Phoenix Dataset upload.

Rewrite of ``evaluation/question_generation.py``.  Preserves async
fanout + semaphore pattern.  Outputs directly to a Phoenix Dataset
(not JSON/pickle).

Top-level imports: stdlib only.  Heavy deps imported lazily.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

logger = logging.getLogger(__name__)


async def _invoke_for_chunk(
    llm,
    messages: list,
    parent_id: str,
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    """Call the LLM for a single parent chunk (with concurrency control).

    Returns a list of QA pair dicts ``[{question, ground_truth}, ...]``.
    """
    async with semaphore:
        t0 = time.time()
        try:
            logger.info("Generating QA for parent ID: %s", parent_id)
            response = await llm.ainvoke(messages)
            data = json.loads(response.content)

            # Handle both raw list and wrapped {items: [...]} shapes
            from rag_core.schemas import SyntheticsTestSet  # lazy

            parsed = SyntheticsTestSet.model_validate(data)
            pairs = [
                {
                    "question": item.question,
                    "ground_truth": item.ground_truth,
                }
                for item in parsed.items
                if hasattr(item, "question") and hasattr(item, "ground_truth")
            ]
            logger.info(
                "Completed parent %s in %.2fs (%d pairs)",
                parent_id,
                time.time() - t0,
                len(pairs),
            )
            return pairs
        except Exception as exc:
            logger.warning(
                "Skipped parent %s (%.2fs): %s",
                parent_id,
                time.time() - t0,
                exc,
            )
            return []


async def _generate_all(config: dict, eval_config) -> list[dict]:
    """Generate QA pairs from all parent chunks in parallel."""
    import pickle

    from rag_core.agents import AgentFactory
    from rag_core.prompts import AgentPrompt
    from rag_core.schemas import SyntheticsTestSet

    store_path = config.get("DOCUMENT_STORE_PERSIST_PATH", "")
    if not store_path:
        raise ValueError("DOCUMENT_STORE_PERSIST_PATH not configured")

    with open(store_path, "rb") as fh:
        document_store = pickle.load(fh)
    logger.info("Loaded %d parent docs from %s", len(document_store), store_path)

    schema = SyntheticsTestSet.model_json_schema()
    json_schema = json.dumps(schema)
    prompt_template = AgentPrompt.from_package("evaluation_qa_generator").chat_template

    llm = AgentFactory.get_judge_model(
        provider=eval_config.judge_provider,
        config=config,
    )
    semaphore = asyncio.Semaphore(eval_config.max_workers)

    tasks = []
    for parent_id, md_doc in document_store.items():
        content = md_doc["content"] if isinstance(md_doc, dict) else md_doc
        if len(content) < 300:
            continue

        prompt = prompt_template.invoke(
            {"json_schema": json_schema, "document_text": content}
        )
        messages = prompt.to_messages()
        tasks.append(_invoke_for_chunk(llm, messages, parent_id, semaphore))

    t0 = time.time()
    results = await asyncio.gather(*tasks)
    all_pairs = [pair for chunk_pairs in results for pair in chunk_pairs]
    logger.info(
        "Generated %d QA pairs from %d chunks in %.2fs",
        len(all_pairs),
        len(results),
        time.time() - t0,
    )
    return all_pairs


async def generate_synthetic_dataset(config: dict, eval_config) -> str:
    """Generate synthetic QA pairs and upload directly to Phoenix Dataset.

    Returns the Phoenix Dataset name.
    """
    qa_pairs = await _generate_all(config, eval_config)

    if not qa_pairs:
        logger.warning("No QA pairs generated — cannot upload empty dataset")
        return ""

    import pandas as pd  # lazy
    from evaluation.phoenix_bootstrap import get_phoenix_client

    dataset_name = f"rag-synthetic-v{int(time.time())}"
    df = pd.DataFrame(qa_pairs)
    df = df.rename(columns={"question": "input", "ground_truth": "expected"})

    client = get_phoenix_client()
    client.datasets.create_dataset(
        name=dataset_name,
        dataframe=df,
        input_keys=["input"],
        output_keys=["expected"],
    )

    logger.info("Synthetic dataset '%s' uploaded (%d pairs)", dataset_name, len(qa_pairs))
    return dataset_name
