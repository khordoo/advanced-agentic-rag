from __future__ import annotations

import importlib
import json
from functools import cached_property
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar

import yaml
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, ConfigDict, PrivateAttr

from rag_core.prompts.profile_config import resolve_prompt_profile_context

_PROMPT_PACKAGES_DIR = Path(__file__).resolve().parent / "packages"


class PromptTemplateConfig(BaseModel):
    """Raw YAML config for a prompt package."""

    name: str
    description: str | None = None
    temperature: float = 0.2
    system_template: str | None = None
    user_template: str


class AgentPrompt(BaseModel):
    """Compiled prompt package (template + optional examples/schema)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    package_id: str
    name: str
    description: str | None = None
    temperature: float = 0.2
    system_template: str | None = None
    user_template: str

    _examples: list[Any] | None = PrivateAttr(default=None)
    _schema_class: type[BaseModel] | None = PrivateAttr(default=None)

    _cache: ClassVar[dict[str, "AgentPrompt"]] = {}

    @classmethod
    def from_package(cls, package_id: str) -> "AgentPrompt":
        if package_id in cls._cache:
            return cls._cache[package_id]

        package_dir = _PROMPT_PACKAGES_DIR / package_id
        template_path = package_dir / "template.yaml"
        if not template_path.exists() or not template_path.is_file():
            raise FileNotFoundError(f"Prompt package template not found at {template_path}")

        raw = yaml.safe_load(template_path.read_text(encoding="utf-8"))
        template_config = PromptTemplateConfig.model_validate(raw)

        prompt = cls(package_id=package_id, **template_config.model_dump())
        prompt._examples = cls._load_examples(package_id)
        prompt._schema_class = cls._load_schema(package_id)

        cls._cache[package_id] = prompt
        return prompt

    @staticmethod
    def _import_optional_module(module_name: str) -> ModuleType | None:
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            # Optional module is considered absent only when the missing module
            # is the requested module itself (e.g., no examples.py file).
            if exc.name == module_name:
                return None
            raise

    @classmethod
    def _load_examples(cls, package_id: str) -> list[Any] | None:
        module_name = f"rag_core.prompts.packages.{package_id}.examples"
        module = cls._import_optional_module(module_name)
        if module is None:
            return None

        if not hasattr(module, "EXAMPLES"):
            raise RuntimeError(
                f"Prompt package '{package_id}' has examples.py but missing EXAMPLES"
            )
        return getattr(module, "EXAMPLES")

    @classmethod
    def _load_schema(cls, package_id: str) -> type[BaseModel] | None:
        module_name = f"rag_core.prompts.packages.{package_id}.schema"
        module = cls._import_optional_module(module_name)
        if module is None:
            return None

        if not hasattr(module, "OutputSchema"):
            raise RuntimeError(
                f"Prompt package '{package_id}' has schema.py but missing OutputSchema"
            )

        schema_class = getattr(module, "OutputSchema")
        if not isinstance(schema_class, type) or not issubclass(schema_class, BaseModel):
            raise RuntimeError(
                f"Prompt package '{package_id}' OutputSchema must be a Pydantic BaseModel"
            )
        return schema_class

    @cached_property
    def chat_template(self) -> ChatPromptTemplate:
        messages = []
        if self.system_template:
            messages.append(("system", self.system_template))
        messages.append(("user", self.user_template))

        base_template = ChatPromptTemplate.from_messages(messages)
        partials_to_inject: dict[str, str] = {}

        if self._examples is not None:
            partials_to_inject["examples"] = self._sanitize_examples(self._examples)

        if self._schema_class is not None:
            parser = PydanticOutputParser(pydantic_object=self._schema_class)
            partials_to_inject["schema"] = parser.get_format_instructions()

        partials_to_inject.update(resolve_prompt_profile_context(self.package_id))

        rendered_template = "\n".join(
            segment for segment in [self.system_template, self.user_template] if segment
        )
        if "{examples}" in rendered_template and "examples" not in partials_to_inject:
            raise RuntimeError(
                f"Prompt package '{self.package_id}' references '{{examples}}' "
                "but has no examples.py"
            )
        if "{schema}" in rendered_template and "schema" not in partials_to_inject:
            raise RuntimeError(
                f"Prompt package '{self.package_id}' references '{{schema}}' "
                "but has no schema.py"
            )

        if partials_to_inject:
            return base_template.partial(**partials_to_inject)
        return base_template

    @staticmethod
    def _sanitize_examples(examples: list[Any]) -> str:
        if not examples:
            return ""

        chunks = []
        for i, example in enumerate(examples, start=1):
            payload = example.model_dump() if isinstance(example, BaseModel) else example
            chunks.append(
                f"--- EXAMPLE {i} ---\n"
                + json.dumps(payload, ensure_ascii=True, indent=2)
            )

        return "\n\n".join(chunks)
