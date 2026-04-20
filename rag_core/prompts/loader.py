from pydantic import BaseModel
from typing import Optional
from functools import cached_property
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pathlib import Path
import importlib
import yaml
import json
from typing import ClassVar

from rag_core.prompts.specs.base import RagFinalResponseSchema

_PROMPTS_DIR = Path(__file__).resolve().parent / "templates"


class AgentPrompt(BaseModel):
    """Parsed YAML prompt template."""

    name: str
    description: Optional[str] = None
    temperature: float = 0.2
    examples_var: str | None = None
    schema_class: str | None = None
    system_template: str | None = None
    user_template: str
    _cache: ClassVar[dict[str, "AgentPrompt"]] = {}

    @classmethod
    def from_yaml(cls, name: str) -> "AgentPrompt":
        if name in cls._cache:
            return cls._cache[name]
        path = _PROMPTS_DIR / f"{name}.yaml"
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Prompt not found at {path}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        cls._cache[name] = cls(**raw)
        return cls._cache[name]

    @cached_property
    def chat_template(self) -> ChatPromptTemplate:
        messages = []
        if self.system_template:
                messages.append(("system", self.system_template))
        messages.append(("user", self.user_template))

        base_template = ChatPromptTemplate.from_messages(messages)

        partials_to_inject = {}

        if self.examples_var:
            try:
                module_path, var_name = self.examples_var.rsplit('.', 1)
                module = importlib.import_module(module_path)
                examples_list = getattr(module, var_name)
                partials_to_inject['examples'] = self._sanitize_examples(examples_list)
            except Exception as e:
                raise RuntimeError(f"Failed to load examples from {self.examples_var}: {e}")

        if self.schema_class:
            try:
                module_path, class_name = self.schema_class.rsplit('.', 1)
                module = importlib.import_module(module_path)
                schema_class = getattr(module, class_name)
                # Use LangChain's parser to get the clean instructions
                parser = PydanticOutputParser(pydantic_object=schema_class)
                partials_to_inject['schema'] = parser.get_format_instructions()
            except Exception as e:
                raise RuntimeError(f"Failed to load schema: {e}")
        if partials_to_inject:
            return base_template.partial(**partials_to_inject)
        
        return base_template

    def _sanitize_examples(self,examples:list[RagFinalResponseSchema]):
        """Private method: Converts a list of Pydantic objects into a clean JSON string."""
        if not examples:
            return ""
        raw = '\n\n'.join(
                [  f'--- EXAMPLE {i+1} ---\n' + json.dumps(e.model_dump(),indent=2)
                for i, e in enumerate(examples)
                ])
        return raw
