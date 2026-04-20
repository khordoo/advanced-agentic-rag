"""Singleton manager for Qdrant client connections with FastEmbed support."""

from __future__ import annotations

import logging
from threading import Lock

from qdrant_client import QdrantClient
from tenacity import retry, stop_after_attempt, wait_exponential

from rag_core.core.config import QdrantSettings, get_qdrant_settings
from rag_core.exceptions import VectorDatabseConnectionError

logger = logging.getLogger(__name__)


class QdrantConnectionManager:
    """Manages a thread-safe singleton QdrantClient instance.

    Ensures models are loaded into memory and connection is verified via ping
    before returning the client.
    """

    _client: QdrantClient | None = None
    _lock: Lock = Lock()

    @staticmethod
    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _connect_and_ping(settings: QdrantSettings) -> QdrantClient:
        """Establish connection and verify health before loading models."""
        logger.info("Attempting to connect to Qdrant at %s …", settings.url)
        client = QdrantClient(url=settings.url)

        # Ping: triggers retry logic if DB is unreachable
        client.get_collections()

        logger.info("Loading FastEmbed models into memory …")
        client.set_model(settings.embed_model)
        client.set_sparse_model(settings.sparse_model)

        return client

    @classmethod
    def get_client(cls, settings: QdrantSettings | None = None) -> QdrantClient:
        """Get or initialize the singleton QdrantClient.

        Parameters
        ----------
        settings:
            Optional settings to override defaults. Only used during initialization.

        Returns
        -------
        QdrantClient:
            The initialized and verified Qdrant client.
        """
        if cls._client is not None:
            return cls._client

        active_settings = settings or get_qdrant_settings()

        with cls._lock:
            if cls._client is None:
                try:
                    cls._client = cls._connect_and_ping(active_settings)
                    logger.info("Qdrant client initialization complete.")
                except Exception as e:
                    logger.error("Failed to initialize Qdrant client: %s", e)
                    raise VectorDatabseConnectionError(
                        f"Could not connect to Qdrant at {active_settings.url} "
                        "after multiple retries. Verify Docker services are running."
                    ) from e

        return cls._client
