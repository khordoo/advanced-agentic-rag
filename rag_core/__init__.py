"""
advanced-rag: A production-grade RAG pipeline with hierarchical chunking,
hybrid search (BM25 + vector), cross-encoder reranking, and RAGAS evaluation.
"""

from rag_core.agents import AgentFactory
from rag_core.prompts import AgentPrompt
from rag_core.schemas import PromptConfig, QAPair, QARecord, SyntheticsTestSet

try:
    from rag_core.retriever import Retriever
except ModuleNotFoundError:  # pragma: no cover - optional dependency path
    Retriever = None

__all__ = [
    "convert_pdf_to_markdown",
    "chunk_markdown",
    "Retriever",
    "RAGPipeline",
    "AgentFactory",
    "AgentPrompt",
    "QAPair",
    "SyntheticsTestSet",
    "QARecord",
    "PromptConfig",
]
