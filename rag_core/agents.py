"""LLM provider factory and prompt template loader."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
from langgraph.graph.state import NoneType
import yaml
import json
from pydantic import ValidationError
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_aws import ChatBedrockConverse

from langchain_core.language_models import BaseChatModel
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from rag_core.schemas import PromptConfig, ExtractedTable, TableSummary
from rag_core.prompts.loader import AgentPrompt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent / LLM factory
# ---------------------------------------------------------------------------


class AgentFactory:
    """Create LangChain chat-model instances by provider name."""

    DEFAULT_TEMP = 0.2
    DEFAULT_AWS_MODEL = "amazon.nova-micro-v1:0"
    DEFAULT_MODEL = "amazon.nova-micro-v1:0"
    DEFAULT_KIMI_MODEL = "moonshotai/kimi-k2.5"
    DEFAULT_SUMMERIZER_MODEL = "lmstudio-community/gemma-4-e2b-it:Q4_K_S"

    @classmethod
    def create(
        cls,
        provider: str,
        temperature: float = DEFAULT_TEMP,
        model_name: str | None = None,
        config: dict | None = None,
    ) -> BaseChatModel:
        """Return a chat model for the given *provider*.

        Parameters
        ----------
        provider:
            One of ``aws``, ``lmstudio``, ``ollama``, ``kimi``.
        temperature:
            Sampling temperature.
        model_name:
            Override the default model for the provider.
        config:
            Optional config dict (from :func:`rag_core.config.load_config`).
            Required for providers that need endpoint URLs / API keys.
        """
        config = config or {}
        provider = provider.lower().strip()

        builders: dict[str, callable] = {
            "aws": lambda: ChatBedrockConverse(
                model=model_name or cls.DEFAULT_AWS_MODEL,
                region_name="us-east-1",
                temperature=temperature,
            ),
            "lmstudio": lambda: ChatOpenAI(
                model=model_name or "dummy",
                base_url=config.get(
                    "OPEN_AI_COMPATIBLE_BASE_URL", "http://127.0.0.1:1234/v1"
                ),
                temperature=temperature,
            ),
            "ollama": lambda: ChatOllama(
                model=model_name or cls.DEFAULT_OLLAMA_MODEL,
                base_url=config.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
                temperature=temperature,
            ),
            "kimi": lambda: ChatOpenAI(
                model=model_name or cls.DEFAULT_KIMI_MODEL,
                base_url=config.get("NVIDIA_KIMI_K2_BASE_URL", ""),
                api_key=config.get("NVIDIA_KIMI_API_KEY", ""),
                temperature=temperature,
            ),
        }

        if provider not in builders:
            raise ValueError(
                f"Unknown provider: '{provider}'. "
                f"Available: {', '.join(sorted(builders))}"
            )

        return builders[provider]()

    @classmethod
    def get_judge_model(
        cls,
        provider: str = "aws",
        config: dict | None = None,
    ) -> BaseChatModel:
        """Return a deterministic model suitable for LLM-as-judge evaluation."""
        return cls.create(
            provider,
            model_name="amazon.nova-pro-v1:0",
            temperature=0,
            config=config,
        )

    @classmethod
    def get_default_model(
        cls,
        provider: str = "aws",
        model_name: str | None = None,
        config: dict | None = None,
    ) -> BaseChatModel:
        """Return a deterministic model suitable for LLM-as-judge evaluation."""
        return cls.create(
            provider,
            model_name=model_name or cls.DEFAULT_MODEL,
            temperature=0,
            config=config,
        )

    @classmethod
    def get_table_summerizer(cls, provider="lmstudio", prompt_name="table_summary"):
        agent = cls.create(provider=provider)
        prompt = AgentPrompt.from_package(prompt_name)
        return TableSummarizer(agent, prompt)


def log_retry(retry_state):
    print(
        f"attempt={retry_state.attempt_number} "
        f"failed with={retry_state.outcome.exception()!r}"
    )


class TableSummarizer:
    """A specialized agent for turning complex tables into searchable summaries."""

    def __init__(
        self,
        model: Optional[BaseChatModel] = None,
        prompt: AgentPrompt | None = None,
    ):
        self.llm = model or AgentFactory.create(provider="aws")
        # Load our 'table_summary' prompt once
        self.prompt = prompt or AgentPrompt.from_package("table_summary")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(
            (
                json.JSONDecodeError,
                ValidationError,
            )
        ),
        before_sleep=log_retry,
        reraise=True,
    )
    def summarize(self, table_markdown: str) -> str:
        """Generates a concise summary for a single table."""
        print("starting summarization")
        logger.info(
            "Generating table summary", extra={"table_chars": len(table_markdown)}
        )
        messages = self.prompt.chat_template.invoke({"table_markdown": table_markdown})
        print("===Tables messages:", messages.to_messages())
        logger.debug(
            "Rendered table summary prompt",
            extra={"prompt_preview": str(messages.to_messages()[0].content[:200])},
        )
        print("===Tables messages:", messages.to_messages())
        res = self.llm.invoke(messages.to_messages())
        logger.debug("LLM returned response", extra={"raw_response": res.content})
        data = json.loads(res.content)
        parsed = TableSummary.model_validate(data)
        logger.info("Table summary generated successfully")
        return parsed.to_markdown()
