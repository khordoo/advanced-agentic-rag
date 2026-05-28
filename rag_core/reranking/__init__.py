from rag_core.reranking.base import BaseReranker

__all__ = ["BaseReranker", "BedrockReranker", "FastEmbedReranker", "get_reranker"]


def __getattr__(name: str):
    if name == "FastEmbedReranker":
        from rag_core.reranking.fastembed_reranker import FastEmbedReranker

        return FastEmbedReranker
    if name == "BedrockReranker":
        from rag_core.reranking.bedrock_reranker import BedrockReranker

        return BedrockReranker
    if name == "get_reranker":
        from rag_core.reranking.factory import get_reranker

        return get_reranker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
