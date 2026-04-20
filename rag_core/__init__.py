"""
advanced-rag: A production-grade RAG pipeline with hierarchical chunking,
hybrid search (BM25 + vector), cross-encoder reranking, and RAGAS evaluation.
"""

from rag_core.converter import convert_pdf_to_markdown
from rag_core.chunker import chunk_markdown
from rag_core.retriever import Retriever
from rag_core.pipelines.pipeline import RAGPipeline
from rag_core.agents import AgentFactory
from rag_core.prompts import AgentPrompt
from rag_core.schemas import QAPair, SyntheticsTestSet, QARecord, PromptConfig

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
