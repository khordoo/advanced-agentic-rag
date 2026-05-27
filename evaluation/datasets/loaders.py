"""CSV→Phoenix Dataset upload with content-hash idempotency.

Top-level imports: stdlib only.  ``phoenix`` and ``pandas`` are imported
lazily so ``import evaluation.datasets.loaders`` is safe on a fresh clone.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _dataset_has_hash(dataset, content_hash: str, expected_count: int) -> bool:
    """Return True when every existing example carries the dataset hash."""
    examples = getattr(dataset, "examples", {}) or {}
    if len(examples) != expected_count:
        return False

    example_values = examples.values() if hasattr(examples, "values") else examples
    for example in example_values:
        metadata = getattr(example, "metadata", {}) or {}
        if isinstance(example, dict):
            metadata = example.get("metadata", {}) or {}
        if metadata.get("content_sha256") != content_hash:
            return False
    return True


def load_csv_rows(path: str | Path) -> list[dict]:
    """Read *ground_truth_dataset.csv* and return a list of dicts (one per row).

    Expected columns: ``question``, ``ground_truth``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Golden dataset CSV not found: {path}")

    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            # Skip empty rows
            if not any(v.strip() for v in row.values()):
                continue
            rows.append(dict(row))

    logger.info("Loaded %d rows from %s", len(rows), path)
    return rows


def compute_dataset_hash(rows: list[dict]) -> str:
    """Compute a deterministic SHA-256 over normalised rows.

    Rows are sorted by JSON key-order then by content to ensure
    identical datasets always produce the same hash regardless of
    row ordering in the CSV.
    """
    canonical = json.dumps(
        sorted(rows, key=lambda r: json.dumps(r, sort_keys=True)),
        sort_keys=True,
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def ensure_dataset_uploaded(eval_config) -> str:
    """Upload the golden CSV to Phoenix as a Dataset (idempotent).

    Returns the Phoenix dataset ID/name handle.

    Idempotency: skips upload if Phoenix already has a dataset with the
    same ``dataset_name`` AND matching ``content_sha256`` metadata.
    """
    from evaluation.phoenix_bootstrap import get_phoenix_client

    rows = load_csv_rows(eval_config.dataset_csv_path)

    # Cap examples before hashing/upload so idempotency matches what Phoenix stores.
    if eval_config.max_examples and len(rows) > eval_config.max_examples:
        logger.info(
            "Capping dataset from %d to %d examples (EVAL_MAX_EXAMPLES)",
            len(rows),
            eval_config.max_examples,
        )
        rows = rows[: eval_config.max_examples]

    content_hash = compute_dataset_hash(rows)
    dataset_name = eval_config.dataset_name

    client = get_phoenix_client()

    # Check if dataset already exists with the same hash
    try:
        existing_ds = client.datasets.get_dataset(dataset=dataset_name)
        if existing_ds is not None:
            if _dataset_has_hash(existing_ds, content_hash, expected_count=len(rows)):
                logger.info(
                    "Dataset '%s' already up-to-date (hash=%s…) — skipping upload",
                    dataset_name,
                    content_hash[:12],
                )
                return dataset_name
            logger.info(
                "Dataset '%s' exists but hash differs — uploading new version",
                dataset_name,
            )
    except Exception:
        logger.info("Dataset '%s' not found — creating", dataset_name)

    # Prepare examples for Phoenix Dataset upload
    import pandas as pd  # lazy — only needed for upload

    df = pd.DataFrame(rows)

    # Rename columns to match Phoenix expected schema
    column_map = {}
    if "question" in df.columns:
        column_map["question"] = "input"
    if "ground_truth" in df.columns:
        column_map["ground_truth"] = "expected"
    df = df.rename(columns=column_map)
    df["content_sha256"] = content_hash

    # Upload to Phoenix
    client.datasets.create_dataset(
        name=dataset_name,
        dataframe=df,
        input_keys=["input"],
        output_keys=["expected"],
        metadata_keys=["content_sha256"],
    )

    logger.info(
        "Dataset '%s' uploaded to Phoenix (%d examples, hash=%s…)",
        dataset_name,
        len(df),
        content_hash[:12],
    )
    return dataset_name
