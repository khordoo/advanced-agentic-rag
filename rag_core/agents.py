"""LLM provider factory and prompt template loader."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from langchain_aws import ChatBedrockConverse
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from rag_core.schemas import PromptConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt loader
# ---------------------------------------------------------------------------

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


class PromptLoader:
    """Load and render YAML prompt templates shipped with the package."""

    @classmethod
    def load(cls, name: str, **kwargs) -> PromptConfig:
        """Load a prompt by *name* (without extension) and render it.

        Any extra ``**kwargs`` are interpolated into the template string via
        ``str.format()``.
        """
        file_path = _PROMPTS_DIR / f"{name}.yaml"
        if not file_path.is_file():
            raise FileNotFoundError(f"Prompt template not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as fh:
            raw_data = yaml.safe_load(fh)

        config = PromptConfig(**raw_data)
        config.template = config.template.format(**kwargs)
        return config


# ---------------------------------------------------------------------------
# Agent / LLM factory
# ---------------------------------------------------------------------------


class AgentFactory:
    """Create LangChain chat-model instances by provider name."""

    DEFAULT_TEMP = 0.2
    DEFAULT_AWS_MODEL = "amazon.nova-micro-v1:0"
    DEFAULT_KIMI_MODEL = "moonshotai/kimi-k2.5"
    DEFAULT_OLLAMA_MODEL = "gemma4:e2b"

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
                base_url=config.get("OPEN_AI_COMPATIBLE_BASE_URL", "http://127.0.0.1:1234/v1"),
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
