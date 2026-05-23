"""
advanced-rag: A production-grade RAG pipeline with hierarchical chunking,
hybrid search (BM25 + vector), cross-encoder reranking, and RAGAS evaluation.
"""

from rag_core.agents import AgentFactory
from rag_core.prompts import AgentPrompt
from rag_core.schemas import PromptConfig, QAPair, QARecord, SyntheticsTestSet

__all__ = [
    "convert_pdf_to_markdown",
    "chunk_markdown",
    "RAGPipeline",
    "AgentFactory",
    "AgentPrompt",
    "QAPair",
    "SyntheticsTestSet",
    "QARecord",
    "PromptConfig",
]
