# Ingestion

The ingestion input root is `data/source_docs/`.

## Supported Input Files

- `.pdf`
- `.md`

## Folder-to-Tag Mapping

The immediate parent folder name is mapped to the document tag.

Examples:
- `data/source_docs/finance/report.pdf` -> `finance`
- `data/source_docs/hr/policy.md` -> `hr`

## Files and Folders Ignored by Ingestion

Ignored folders (not traversed):
- `.DS_Store`
- `__pycache__`
- `.git`

Ignored files:
- `README.md`
- `LICENSE`
- `.gitignore`

This is why guidance docs should live in `docs/` or as `README.md` files inside `data/source_docs/`.
