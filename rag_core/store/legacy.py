from typing import Optional
import functools
import logging
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    """Factory for initializing vector stores with optional shared embeddings.

    This class centralizes the creation of different vector store backends
    (Chroma, InMemory) and ensures that heavy embedding models are cached
    and reused efficiently across the application.
    """

    @classmethod
    def from_chroma(cls, config: dict, embeddings: Optional[Embeddings] = None):
        """Initialize a persisted Chroma vector store.

        Args:
            config: Configuration dictionary containing 'CHROMA_PERSIST_DIRECTORY',
                'CHROMA_COLLECTION_NAME', and 'HF_EMBEDDING_MODEL'.
            embeddings: Optional pre-initialized embedding object. If None,
                a new model will be loaded based on the config.

        Returns:
            Chroma: A persisted LangChain-compatible vector store instance.
        """
        from langchain_chroma import Chroma
        if embeddings is None:
            model_name = config.get("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            logger.info(f"Loading embedding model: {model_name}")
            embeddings = cls._get_embeddings(model_name)
        return Chroma(
            collection_name=config.get("CHROMA_COLLECTION_NAME", "docs"),
            embedding_function=embeddings,
            persist_directory=config.get("CHROMA_PERSIST_DIRECTORY", "artifacts/chroma_db"),
        )

    @classmethod
    def from_inmemory(cls, config: dict, embeddings: Optional[Embeddings] = None):
        """Initialize an ephemeral in-memory vector store for testing or small datasets.

        Args:
            config: Configuration dictionary for fallback embedding model name.
            embeddings: Optional pre-initialized embedding object.

        Returns:
            InMemoryVectorStore: A non-persisted vector store.
        """
        if embeddings is None:
            model_name = config.get("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            logger.info(f"Loading embedding model: {model_name}")
            embeddings = cls._get_embeddings(model_name)
        return InMemoryVectorStore(embeddings)

    @classmethod
    @functools.lru_cache
    def _get_embeddings(cls, model_name: str):
        """Load and cache HuggingFace embedding models.

        This method uses @lru_cache to ensure that the same model is only
        loaded into memory once, even if requested multiple times.

        Args:
            model_name: The HuggingFace model identifier string.

        Returns:
            HuggingFaceEmbeddings: The initialized and cached embedding object.
        """
        logger.debug(f"Cache miss: Fetching model '{model_name}' from HuggingFace")
        return HuggingFaceEmbeddings(model_name=model_name)

class DocumentStore:
    @classmethod
    def from_dict(cls):
        return {}
    