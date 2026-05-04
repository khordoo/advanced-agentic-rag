# Ingestion

The ingestion input root is `data/source_docs/`.

## Metadata Extension Hooks

Metadata extraction lives in `rag_core/ingestion/metadata.py`.

The pipeline calls `extract_metadata_from_uri(source_uri)`, which delegates to these intentional extension hooks:
- `_extract_scopes(path_obj)`
- `_extract_tags(path_obj)`
- `_extract_extended_metadata(path_obj)`

These hooks are meant to be overridden for project-specific rules (for example: enterprise scope policies, custom tags, or additional JSON metadata).

Recommended contract:
- Keep hook function signatures unchanged.
- Keep return types stable:
  - `list[str]` for scopes and tags
  - `dict[str, object]` for extended metadata
- Keep `extract_metadata_from_uri(...)` as the stable entrypoint used by ingestion.

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
