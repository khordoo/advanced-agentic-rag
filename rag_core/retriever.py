"""Retriever with hierarchical parent-child search and cross-encoder reranking."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class Retriever:
    """Hierarchical retriever: search child chunks → deduplicate → return parents.

    Parameters
    ----------
    vector_store:
        Any LangChain-compatible vector store that has a
        ``similarity_search(query, k=...)`` method.
    doc_store:
        Parent document store — ``dict[str, dict]`` with
        ``{"content": ..., "metadata": ...}`` values.
    config:
        Configuration dict (from :func:`rag_core.config.load_config`).
    """

    def __init__(
        self,
        vector_store,
        doc_store: dict[str, dict],
        config: dict | None = None,
    ) -> None:
        self.config = config or {}
        self.vector_store = vector_store
        self.doc_store = doc_store
        self.reranker = CrossEncoder(
            self.config.get("HF_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
        )

    # -- convenience constructors ------------------------------------------

    @classmethod
    def from_chroma(cls, config: dict) -> "Retriever":
        """Build a retriever backed by a persisted Chroma store."""
        from langchain_chroma import Chroma

        emb = HuggingFaceEmbeddings(
            model_name=config.get("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        )
        vector_store = Chroma(
            collection_name=config.get("CHROMA_COLLECTION_NAME", "docs"),
            embedding_function=emb,
            persist_directory=config.get("CHROMA_PERSIST_DIRECTORY", "artifacts/chroma_db"),
        )

        doc_store: dict = {}
        store_path = config.get("DOCUMENT_STORE_PERSIST_PATH", "")
        if store_path and Path(store_path).is_file():
            with open(store_path, "rb") as fh:
                doc_store = pickle.load(fh)
            logger.info("Loaded document store from %s", store_path)
        else:
            logger.warning("Document store not found at %s", store_path)

        return cls(vector_store=vector_store, doc_store=doc_store, config=config)

    # -- search methods ----------------------------------------------------

    def similarity_search(self, query: str, k: int = 10):
        """Raw vector similarity search."""
        return self.vector_store.similarity_search(query, k=k)

    def rerank(self, query: str, docs, top_n: int | None = None):
        """Re-score documents with a cross-encoder and return the top-N."""
        top_n = top_n or int(self.config.get("RERANK_TOP_N", 3))
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.reranker.predict(pairs)
        scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        ranked = [doc for _, doc in scored[:top_n]]
        logger.info(
            "Reranked %d docs → top %d (scores: %s)",
            len(docs),
            len(ranked),
            [f"{s:.3f}" for s, _ in scored[:top_n]],
        )
        return ranked

    def hierarchical_search(self, query: str, k: int = 15, top_n: int = 3) -> list[str]:
        """Search child chunks → rerank → deduplicate → return parent contents.

        This is the primary retrieval method used by :class:`RAGPipeline`.
        """
        child_docs = self.similarity_search(query, k=k)
        child_docs = self.rerank(query, child_docs, top_n=top_n)

        seen_parents: set[str] = set()
        parent_contents: list[str] = []

        for doc in child_docs:
            parent_id = doc.metadata.get("parent")
            if parent_id and parent_id not in seen_parents:
                seen_parents.add(parent_id)
                parent = self.doc_store.get(parent_id)
                if parent:
                    parent_contents.append(parent.get("content", ""))
                else:
                    logger.warning(
                        "Parent ID %s found in vector DB but missing in document store",
                        parent_id,
                    )

        logger.info(
            "Hierarchical search: %d children → %d unique parents",
            len(child_docs),
            len(parent_contents),
        )
        return parent_contents
