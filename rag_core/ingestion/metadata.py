"""Metadata extraction helpers for ingestion.

Extension policy:
- `extract_metadata_from_uri()` is the stable entrypoint used by ingestion.
- `_extract_scopes`, `_extract_tags`, and `_extract_extended_metadata` are
  intentional customization hooks for project-specific rules.
- Keep hook signatures stable if you modify them.
"""

from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse
import logging
import os

logger = logging.getLogger(__name__)

@dataclass
class AssetMetadata:
    file_name: str
    folder_name: str
    extension: str
    tags: list[str] = field(default_factory=list)
    scopes: list[str] = field(default_factory=list)
    extended_metadata: dict[str, object] = field(default_factory=dict)

# ==========================================
# EXTENSION POINTS: MODIFY THESE FUNCTIONS
# ==========================================

def _extract_scopes(path_obj: Path) -> list[str]:
    """
    HOOK: Override to define Security Scopes.
    Defaults to 'public'. Explicitly restricted folders can override this.
    """
    # Start with the baseline security posture
    scopes = ["public"]

    # (Optional Future Logic) Override if it's in a sensitive directory
    if "finance" in path_obj.parts or "confidential" in path_obj.parts:
        # Remove 'public' and apply strict scopes
        scopes = ["finance_dept"]
    return scopes

def _extract_tags(path_obj: Path) -> list[str]:
    """
    HOOK: Override this logic to define how retrieval tags are assigned.
    Uses extension and folder hierarchy for topical routing.
    """
    tags = []
    if path_obj.suffix:
        # e.g., ".pdf" -> "type:pdf"
        clean_ext = path_obj.suffix.lower().replace(".", "")
        tags.append(f"type:{clean_ext}")
    # Tag 2: Topic (Folder Name)
    if path_obj.parent.name:
        # e.g., "folder:onboarding"
        tags.append(f"folder:{path_obj.parent.name.lower()}")
    # TODO: Add custom tagging logic here
    return tags


def _extract_extended_metadata(path_obj: Path) -> dict[str, object]:
    """
    HOOK: Override this to extract arbitrary key-value pairs for the JSONB column.
    Useful for file stats, version parsing, or custom flags.

    """
    metadata = {}

    # Example: If it's a local file, we can easily grab its size
    if path_obj.exists() and path_obj.is_file():
        metadata["file_size_bytes"] = os.path.getsize(path_obj)

    # TODO: Add custom dictionary logic here (e.g., regex matching versions from filename)
    return metadata


# ==========================================
# CORE PARSING LOGIC
# ==========================================

def extract_metadata_from_uri(source_uri: str) -> AssetMetadata:
    """Safely parses URIs and executes metadata hooks."""
    parsed_uri = urlparse(source_uri)
    path_obj = Path(parsed_uri.path)

    # Base Metadata
    file_name = path_obj.name
    folder_name = path_obj.parent.name
    extension = path_obj.suffix.lower()

    # Execute Hooks for Custom Logic
    scopes = _extract_scopes(path_obj)
    tags = _extract_tags(path_obj)
    extended_metadata = _extract_extended_metadata(path_obj)
    return AssetMetadata(
        file_name=file_name,
        folder_name=folder_name,
        extension=extension,
        tags=tags,
        scopes=scopes,
        extended_metadata=extended_metadata
    )
