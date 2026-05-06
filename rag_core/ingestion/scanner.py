# rag_core/ingestion/scanner.py
from typing import Protocol
from pathlib import Path
from typing import Iterator
import logging

from rag_core.store.storage.provider import SourceProvider
from rag_core.core.config import IngestionSettings

logger = logging.getLogger(__name__)

class SourceAssetScanner(Protocol):
    def discover_assets(self, base_uri: str) -> Iterator[str]:
        ...


class FileScanner:
    def __init__(self, provider: SourceProvider, settings: IngestionSettings):
        self.source_provider = provider
        self.settings = settings
        self._IGNORE_FILES = {".DS_Store", "Thumbs.db"}

    def discover_assets(self, base_uri: str) -> Iterator[str]:
        """Applies business rules to the raw file stream from the storage provider."""
        logger.info(f"Discovering files from: {base_uri}")
        for file_uri in self.source_provider.iter_assets(base_uri):
            path_obj = Path(file_uri)
            # Apply Policy 1: Ignore junk files
            if path_obj.name in self._IGNORE_FILES or path_obj.name.startswith("."):
                continue

            # Apply Policy 2: Allowed extensions
            if path_obj.suffix.lower() not in self.settings.allowed_extensions:
                logger.debug("Skipping unsupported file type: %s", path_obj.name)
                continue

            yield file_uri

        logger.info(f"Completed filtering files from: {base_uri}")

class ApiScanner:
    pass

class AWSS3Scanner:
    pass
