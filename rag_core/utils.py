import hashlib
import re
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class Stopwatch:
    """A simple context manager for timing tasks and logging duration."""
    def __init__(self, label: str = 'Task') -> None:
        self.start = 0.0
        self.label = label
        
    def __enter__(self):
         logger.info(f'{self.label} started...')
         self.start = time.perf_counter()
         return self
         
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.perf_counter() - self.start
        if exc_type is None:
            logger.info(f'{self.label} completed in {duration:0.2f} seconds')
        else:
            logger.info(f'{self.label} failed after {duration:0.2f} seconds')


# ---------------------------------------------------------------------------
# File identity helpers
# ---------------------------------------------------------------------------


def compute_content_hash(file_path: str | Path) -> str:
    """Compute SHA-256 hash of file contents for deduplication.

    Reads the file in 8 KiB chunks to avoid loading large PDFs into memory.
    """
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _sanitize_identifier(raw: str) -> str:
    """Clean a raw string into an agent-friendly identifier.

    - Replaces slashes, spaces, and hyphens with underscores
    - Removes parentheses, brackets, and other special characters
    - Collapses consecutive underscores
    - Strips leading/trailing underscores
    """
    # Replace common separators with underscore
    result = re.sub(r"[/\\\s\-]+", "_", raw)
    # Remove unwanted special characters
    result = re.sub(r"[()[\]{}<>!@#$%^&*+=|;:'\",?`~]", "", result)
    # Collapse multiple underscores
    result = re.sub(r"_+", "_", result)
    # Strip leading/trailing underscores
    return result.strip("_")


def generate_file_id(file_path: str | Path, base_dir: str | Path) -> str:
    """Generate a sanitized, unique, agent-friendly file identifier.

    Format::

        {parent_folder}_{stem}_{NNN}.{ext}

    The three-digit salt (``NNN``) is **deterministic** — derived from the
    SHA-256 content hash — so the same file always produces the same
    file_id.  Sanitisation removes special characters and normalises
    separators to underscores so the value works seamlessly as a Qdrant
    filter key and as a natural-looking filename in LLM context.

    Examples
    --------
    >>> generate_file_id("docs/q3_earnings/report.pdf", "docs")
    'q3_earnings_report_472.pdf'          # salt varies by content

    >>> generate_file_id("docs/Annual Report (Draft).pdf", "docs")
    'docs_Annual_Report_Draft_219.pdf'    # special chars cleaned

    Parameters
    ----------
    file_path:
        Absolute or relative path to the source file.
    base_dir:
        The ingestion root directory.  The parent folder component of the
        file_id is taken relative to this directory.
    """
    file_path = Path(file_path)
    base_dir = Path(base_dir)

    # 1. Determine parent folder relative to ingestion base
    try:
        relative = file_path.relative_to(base_dir)
        parent_folder = str(relative.parent)
    except ValueError:
        parent_folder = file_path.parent.name

    # "." means file is directly in base_dir → use base_dir name
    if parent_folder == ".":
        parent_folder = base_dir.name

    # 2. Deterministic salt from content hash
    content_hash = compute_content_hash(file_path)
    salt = int(content_hash[:6], 16) % 1000

    # 3. Build raw identifier: folder + stem + salt
    stem = file_path.stem
    ext = file_path.suffix  # includes the dot, e.g. ".pdf"
    raw = f"{parent_folder}_{stem}_{salt:03d}"

    # 4. Sanitise and append original extension
    return f"{_sanitize_identifier(raw)}{ext}"
