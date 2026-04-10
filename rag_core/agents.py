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
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from rag_core.schemas import PromptConfig, ExtractedTable ,TableSummary

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt loader
# ---------------------------------------------------------------------------

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

class PromptLoader:
    """Load and render YAML prompt templates shipped with the package."""
    def __init__(self,config:PromptConfig)->"PromptLoader":
        self.config = config

    @classmethod
    def load(cls,name:str)->"PromptLoader":
        file_path = Path(_PROMPTS_DIR) / f'{name}.yaml'
        if not file_path.is_file():
            raise FileNotFoundError(f"Prompt template not found: {file_path}")

        with open(file_path,'r',encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)
        prompt_config = PromptConfig(**raw_data)
        return cls(prompt_config)

    def render(self,**kwargs):
        # Making a copy to avoid overwriting the template place holders {}
        return self.config.model_copy(
            update={"template":self.config.template.format(**kwargs)}
        )

# class PromptLoader:
#     """Load and render YAML prompt templates shipped with the package."""
#     prompt_config =None
#     @classmethod
#     def load(cls, name: str, **kwargs) -> PromptConfig:
#         """Load a prompt by *name* (without extension) and render it.

#         Any extra ``**kwargs`` are interpolated into the template string via
#         ``str.format()``.
#         """
#         file_path = _PROMPTS_DIR / f"{name}.yaml"
#         if not file_path.is_file():
#             raise FileNotFoundError(f"Prompt template not found: {file_path}")

#         with open(file_path, "r", encoding="utf-8") as fh:
#             raw_data = yaml.safe_load(fh)
#         print(f'Raw promt for name{name}:',raw_data)
#         config = PromptConfig(**raw_data)
#         config.template = config.template.format(**kwargs)
#         return config

#     @classmethod
#     def load_(cls, name: str, **kwargs) -> PromptConfig:
#         """Load a prompt by *name* (without extension) and render it.

#         Any extra ``**kwargs`` are interpolated into the template string via
#         ``str.format()``.
#         """
#         file_path = _PROMPTS_DIR / f"{name}.yaml"
#         if not file_path.is_file():
#             raise FileNotFoundError(f"Prompt template not found: {file_path}")

#         with open(file_path, "r", encoding="utf-8") as fh:
#             raw_data = yaml.safe_load(fh)
#         print(f'Raw promt for name{name}:',raw_data)
#         cls.prompt_config = PromptConfig(**raw_data)
#         return cls

#     @classmethod
#     def render(cls,**kargs):
#         if not cls.prompt_config:
#             raise ValueError('No promt is loaded. Please Lazy laod the prompt first')
#         cls.prompt_config.template = cls.prompt_config.template.format(**kargs)
#         return cls.prompt_config


# ---------------------------------------------------------------------------
# Agent / LLM factory
# ---------------------------------------------------------------------------


class AgentFactory:
    """Create LangChain chat-model instances by provider name."""

    DEFAULT_TEMP = 0.2
    DEFAULT_AWS_MODEL = "amazon.nova-micro-v1:0"
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
            model_name="amazon.nova-micro-v1:0",
            temperature=0,
            config=config,
        )

    @classmethod
    def get_table_summerizer(cls,provider='lmstudio', prompt_name='table_summary'):
        agent=cls.create(provider=provider)
        prompt_config = PromptLoader.load(prompt_name)
        return TableSummarizer(agent,prompt_config)

def log_retry(retry_state):
    print(
        f"attempt={retry_state.attempt_number} "
        f"failed with={retry_state.outcome.exception()!r}"
    )

class TableSummarizer:
    """A specialized agent for turning complex tables into searchable summaries."""

    def __init__(self, model: Optional[BaseChatModel]=None,prompt_loader:PromptLoader=None):
        self.llm = model or AgentFactory.create(provider='lmstudio')
        # Load our 'table_summary' prompt once
        self.prompt_loader = prompt_loader or PromptLoader.load('table_summary')
    @retry(
        stop = stop_after_attempt(3),
        wait = wait_exponential(multiplier=1,min=1,max=8),
        retry=retry_if_exception_type(
            (json.JSONDecodeError,ValidationError,)
        ),
        before_sleep=log_retry,
        reraise=True
    )
    def summarize(self, table_markdown: str) -> str:
        """Generates a concise summary for a single table."""
        # Interpolate the table into the template
        print('Attempting to generate summary.....')
        prompt =self.prompt_loader.render(table_markdown=table_markdown)
        res = self.llm.invoke(prompt.template)
        print('RAW Table Summary response ==========================================')
        print(res.content)
        data = json.loads(res.content)
        print('response:',data)
        # parsed= TableSummary(**data)
        parsed = TableSummary.model_validate(data)
        print('completeed summery:',parsed)
        print('--'*50)
        return parsed.to_markdown()
