import hashlib
import posixpath
import re
import unicodedata
from pathlib import PurePosixPath
from typing import BinaryIO, ContextManager
from urllib.parse import urlparse, unquote

def _slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    lower = ascii_only.lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", lower)
    collapsed = re.sub(r"_+", "_", cleaned).strip("_")
    return collapsed or "unknown"

def _canonicalize_source_uri(source_uri: str) -> str:
    parsed = urlparse(source_uri)

    # Safely reconstruct the path including the netloc (bucket name, domain, etc.)
    # We keep the scheme too, so file://data and s3://data hash differently.
    netloc_part = parsed.netloc if parsed.netloc else ""
    path_part = parsed.path if parsed.path else ""

    raw_string = f"{parsed.scheme}://{netloc_part}{path_part}"

    decoded = unquote(raw_string).replace("\\", "/")
    squashed = re.sub(r"/{2,}", "/", decoded).strip()
    return posixpath.normpath(squashed)

def generate_asset_key(source_uri: str, limit: int = 8, max_length: int = 255) -> str:
    """Generate a stable, semantic file ID with a short deterministic hash."""
    canonical = _canonicalize_source_uri(source_uri)

    parsed = urlparse(canonical)
    path_obj = PurePosixPath(parsed.path)

    parent_slug = _slugify(path_obj.parent.name)
    stem_slug = _slugify(path_obj.stem or path_obj.name)
    ext = path_obj.suffix.lower().lstrip(".")

    # Hash the FULL canonical URI
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:limit]
    ext_part = f".{ext}" if ext else ""

    # --- GRACEFUL DEGRADATION LOGIC ---

    # Scenario 1: Extreme edge case (max_length is incredibly tiny).
    # We abandon all human-readability and just return as much of the hash as fits.
    if max_length <= limit + len(ext_part):
        return f"{digest}{ext_part}"[:max_length]

    # Calculate room for human-readable parts (reserving 2 chars for underscores)
    available_text_len = max_length - limit - len(ext_part) - 2

    # Scenario 2: There is practically no room for text.
    # We gracefully degrade to just the hash + extension.
    if available_text_len < 2:
        return f"{digest}{ext_part}"

    # Scenario 3: Standard operation.
    # Split available space safely between parent and stem.
    half_text_len = available_text_len // 2
    parent_slug = parent_slug[:half_text_len].strip("_")
    stem_slug = stem_slug[:half_text_len].strip("_")

    # If the slugs evaluated to empty strings (e.g., weird characters were stripped),
    # use ultra-short fallbacks just to maintain the formatting structure.
    parent_slug = parent_slug or "d" # 'd' for directory
    stem_slug = stem_slug or "f"     # 'f' for file

    return f"{parent_slug}_{stem_slug}_{digest}{ext_part}"


# Backward-compatible alias; keep until callers migrate fully.
generate_asset_id = generate_asset_key


def compute_content_hash(stream_manager: ContextManager[BinaryIO], algorithm: str = "sha256"):
    with stream_manager as stream:
        hash_value = hashlib.file_digest(stream, algorithm)
        return hash_value.hexdigest()
