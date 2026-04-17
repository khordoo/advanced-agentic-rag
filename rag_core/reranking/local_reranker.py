from typing import List
import logging
from rag_core.reranking.base import BaseReranker
from rag_core.config import RerankerSettings
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class LocalReranker(BaseReranker):
    def __init__(self, settings: RerankerSettings | None = None):
        self._settings = settings or RerankerSettings()
        self._model = None
        self._load_model()

    def _load_model(self):
        if self._model is None:
            self._model = CrossEncoder(
                self._settings.model_name, device=self._settings.resolve_device
            )
            logger.info(f"Loaded model {self._settings.model_name}")

    def rerank(
        self, query: str, documents: List[Document], top_k: int = None
    ) -> List[Document]:
        logger.info(f"Reranking {len(documents)} documents for query: {query}")
        scores = self._model.predict([(query, doc.page_content) for doc in documents])
        for i, doc in enumerate(documents):
            doc.metadata["rerank_score"] = float(scores[i])

        documents.sort(key=lambda x: x.metadata["rerank_score"], reverse=True)

        actual_top_k = top_k if top_k is not None else self._settings.top_k
        logger.info(f"Reranked {len(documents)} documents to top {actual_top_k}")
        return documents[:actual_top_k]
