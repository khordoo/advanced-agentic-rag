"""Compatibility shims for RAGAS optional provider imports."""

from __future__ import annotations

import sys
from types import ModuleType


def apply_ragas_compat() -> None:
    """Shim removed optional import paths that RAGAS imports eagerly."""
    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return

    module = ModuleType(module_name)

    class ChatVertexAI:  # pragma: no cover - only used if misconfigured
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "VertexAI is not supported in this environment. "
                "Please use AWS, Ollama, LMStudio, or Kimi judges."
            )

    module.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = module
