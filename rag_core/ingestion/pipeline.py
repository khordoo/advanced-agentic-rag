import logging

from rag_core.core.config import  setup_logging
from rag_core.ingestion.services.discovery import run_discovery

setup_logging()
logger = logging.getLogger(__name__)


def run_pipeline():
    logger.info("Starting ingestion run pipeline")
    run_discovery()
if __name__ == "__main__":
    run_pipeline()
