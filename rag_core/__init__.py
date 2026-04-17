"""
advanced-rag: A production-grade RAG pipeline with hierarchical chunking,
hybrid search (BM25 + vector), cross-encoder reranking, and RAGAS evaluation.
"""

from rag_core.converter import convert_pdf_to_markdown
from rag_core.chunker import chunk_markdown
from rag_core.retriever import Retriever
from rag_core.pipeline import RAGPipeline
from rag_core.agents import AgentFactory, PromptLoader
from rag_core.schemas import QAPair, SyntheticsTestSet, QARecord, PromptConfig

__all__ = [
    "convert_pdf_to_markdown",
    "chunk_markdown",
    "Retriever",
    "RAGPipeline",
    "AgentFactory",
    "PromptLoader",
    "QAPair",
    "SyntheticsTestSet",
    "QARecord",
    "PromptConfig",
]
