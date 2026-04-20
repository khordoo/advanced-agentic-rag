import os
import logging

from langfuse import get_client
from langfuse.langchain import CallbackHandler

logger = logging.getLogger(__name__)

# ── Langfuse env-var defaults (dev-only, override via .env in production) ──
# TODO: Remove these hardcoded defaults after testing
_LANGFUSE_DEFAULTS = {
    "LANGFUSE_PUBLIC_KEY": "pk-lf-admin-001",
    "LANGFUSE_SECRET_KEY": "sk-lf-admin-001",
    "LANGFUSE_BASE_URL": "http://localhost:3000",
}

for key, value in _LANGFUSE_DEFAULTS.items():
    os.environ.setdefault(key, value)


def init_langfuse_client():
    """Initialise the Langfuse SDK singleton (powers @observe decorators).

    Must be called **before** any @observe-decorated function runs.
    Reads credentials from environment variables (LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL).
    """
    try:
        client = get_client()
        logger.info("Langfuse SDK singleton initialised (client=%s)", client)
        return client
    except Exception as e:
        logger.warning("Langfuse SDK init failed: %s", e)
        return None


def get_observability_callbacks():
    """Returns a list of LangChain callbacks for observability.

    Currently returns a Langfuse CallbackHandler if credentials are present.
    The CallbackHandler reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY /
    LANGFUSE_BASE_URL from the environment automatically.
    """
    callbacks = []

    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        try:
            langfuse_handler = CallbackHandler()
            callbacks.append(langfuse_handler)
            logger.info("Langfuse LangChain CallbackHandler enabled.")
        except Exception as e:
            logger.warning("Langfuse CallbackHandler init failed: %s", e)

    return callbacks
