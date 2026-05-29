from __future__ import annotations

import logging

from fastembed.rerank.cross_encoder import TextCrossEncoder
from langchain_core.documents import Document

from rag_core.core.config import RerankerSettings
from rag_core.reranking.base import BaseReranker

logger = logging.getLogger(__name__)


class FastEmbedReranker(BaseReranker):
    """Cross-encoder reranker powered by FastEmbed ONNX Runtime.

    Uses FastEmbed-supported ONNX cross-encoder models, without torch or CUDA
    dependencies in the application image.
    """

    def __init__(self, settings: RerankerSettings | None = None) -> None:
        self._settings = settings or RerankerSettings()
        self._model: TextCrossEncoder | None = None

    def _load_model(self) -> TextCrossEncoder:
        if self._model is None:
            self._model = TextCrossEncoder(
                model_name=self._settings.model_name,
                cache_dir=self._settings.fastembed_cache_dir,
            )
            logger.info("Loaded FastEmbed reranker: %s", self._settings.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
    ) -> list[Document]:
        if not documents:
            return []

        model = self._load_model()
        texts = [doc.page_content for doc in documents]
        scores = list(model.rerank(query=query, documents=texts))

        for index, score in enumerate(scores):
            documents[index].metadata["rerank_score"] = float(score)

        documents.sort(key=lambda d: d.metadata.get("rerank_score", 0), reverse=True)
        actual_top_k = top_k if top_k is not None else self._settings.top_k
        return documents[:actual_top_k]
