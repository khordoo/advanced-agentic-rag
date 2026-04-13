from qdrant_client import QdrantClient
import logging
from threading import Lock
from rag_core.config import get_qdrant_settings,QdrantSettings
from rag_core.exceptions import VectorDatabseConnectionError
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class QdrantConnectionManager:
    _client:QdrantClient | None = None
    _lock: Lock = Lock()


    @staticmethod
    @retry(
        stop=stop_after_attempt(4), # Try 4 times total
        wait=wait_exponential(multiplier=1, min=2, max=10), # Wait 2s, 4s, 8s...
        reraise=True # If it fails 4 times, raise the actual underlying error
    )
    def _connect_and_ping(settings: QdrantSettings) -> QdrantClient:
        logger.info(f"Attempting to connect to Qdrant at {settings.url}...")
        client = QdrantClient(url=settings.url)

        # The Ping: This will fail and trigger Tenacity if the DB is down
        client.get_collections()

        # Load the models only upon a successful ping
        client.set_model(settings.embed_model)
        client.set_sparse_model(settings.sparse_model)

        return client

    @classmethod
    def get_client(cls, settings: QdrantSettings | None=None)->QdrantClient:
        """
        Returns a singleton instance of the QdrantClient.
        Thread-safe and lazily initialized.
        """
        if cls._client is not None:
            return cls._client

        active_settings = settings or get_qdrant_settings()

        with cls._lock:
            logger.info("Initializing FastEmbed models into memory...")
            if cls._client is None:

                logger.info(f"Initializing Qdrant client at {active_settings.url}...")
                logger.info("Loading FastEmbed models into memory (this may take a moment)...")
                try:
                    cls._client = cls._connect_and_ping(active_settings)
                    logger.info("Initializing Qdrant client completed.")
                except Exception as e:
                    logger.error(f"Failed to initialize Qdrant client: {e}")
                    raise VectorDatabseConnectionError(
                        f"Could not connect to Qdrant at {active_settings.url} after multiple retries. Is Docker running?"
                    ) from e

        return cls._client
