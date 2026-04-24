# Source Documents for Ingestion

This directory is the ingestion input root for the RAG pipeline.

## How It Works

- Put source files (`.pdf` or `.md`) anywhere under `data/source_docs/`.
- The immediate parent folder name is used as a category tag during ingestion.
- Example: `data/source_docs/finance/q1-report.pdf` gets the tag `finance`.

## Organizing by Category

You can create any category folders you need, for example:

- `finance/`
- `legal/`
- `hr/`
- `technical/`

These folder names are automatically mapped to tags.

## Important

- Markdown guidance files (like this `README.md`) are intentionally ignored.
- Ingestion processes `.pdf` and `.md` files.
