"""Synthetic QA generation from parent document chunks."""

from __future__ import annotations

import asyncio
import json
import logging
import pickle
import time

from rag_core.agents import AgentFactory, PromptLoader
from rag_core.config import load_config
from rag_core.schemas import SyntheticsTestSet

logger = logging.getLogger(__name__)


async def _invoke_for_chunk(
    llm,
    messages: list[dict],
    parent_id: str,
    semaphore: asyncio.Semaphore,
) -> SyntheticsTestSet:
    """Call the LLM for a single parent chunk (with concurrency control)."""
    async with semaphore:
        t0 = time.time()
        try:
            logger.info("Calling API for parent ID: %s", parent_id)
            response = await llm.ainvoke(messages)
            data = json.loads(response.content)
            parsed = SyntheticsTestSet.model_validate(data)
            logger.info(
                "Completed parent %s in %.2fs (%d pairs)",
                parent_id,
                time.time() - t0,
                len(parsed.items),
            )
            return parsed
        except Exception as exc:
            logger.warning(
                "Skipped parent %s (%.2fs): %s",
                parent_id,
                time.time() - t0,
                exc,
            )
            return SyntheticsTestSet(items=[], refusal="Insufficient text")


async def generate_questions_async(config: dict | None = None):
    """Generate synthetic QA pairs from all parent chunks in parallel."""
    config = config or load_config()

    store_path = config["DOCUMENT_STORE_PERSIST_PATH"]
    with open(store_path, "rb") as fh:
        document_store = pickle.load(fh)
    logger.info("Loaded %d parent docs from %s", len(document_store), store_path)

    schema = SyntheticsTestSet.model_json_schema()
    prompt_template = PromptLoader.load("qa_gen", schema=json.dumps(schema), context="")
    system_prompt = prompt_template.template

    llm = AgentFactory.get_judge_model(config=config)
    semaphore = asyncio.Semaphore(int(config.get("LLM_MAX_PARALLEL_REQUEST", 3)))

    tasks = []
    for parent_id, md_doc in document_store.items():
        content = md_doc["content"] if isinstance(md_doc, dict) else md_doc
        if len(content) < 300:
            continue

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Document Texts:\n\n{content}"},
        ]
        tasks.append(_invoke_for_chunk(llm, messages, parent_id, semaphore))

    t0 = time.time()
    results = await asyncio.gather(*tasks)
    logger.info("Completed %d tasks in %.2fs", len(results), time.time() - t0)

    return results


def save_qa_pairs(path: str, qa_pairs: list[dict]) -> None:
    """Persist QA pairs to a JSON file."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(qa_pairs, fh, ensure_ascii=False, indent=2)
    logger.info("Saved %d QA pairs to %s", len(qa_pairs), path)


def run_question_generation(config: dict | None = None):
    """Full pipeline: load docs → generate QA → save."""
    config = config or load_config()
    results = asyncio.run(generate_questions_async(config))

    qa_pairs = [
        qa.model_dump(exclude={"refusal"})
        for r in results
        for qa in r.items
    ]
    save_qa_pairs(config["SYNTHETIC_QA_PERSIST_PATH"], qa_pairs)
    return qa_pairs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_question_generation()
