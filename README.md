# Advanced RAG Pipeline

Production-oriented Retrieval-Augmented Generation pipeline with hierarchical chunking, reranking, hybrid retrieval, and evaluation tooling.

## Start Here

Use this file as the landing page, then go to detailed guides in [`docs/`](docs/README.md).

- [Documentation Index](docs/README.md)
- [Getting Started](docs/getting-started/README.md)
- [Configuration](docs/configuration/README.md)
- [Generation Model Settings](docs/configuration/generation-models.md)
- [Ingestion Rules](docs/ingestion/README.md)
- [Agents & Prompt Profiles](docs/agents/README.md)

## Quick Start

```bash
# Clone and install
git clone <your-repo-url> && cd advanced-rag
uv sync --extra dev

# Configure
cp .env.example .env

# Ingest source files from data/source_docs/
uv run rag_core/pipelines/ingestion.py

# Start CLI
uv run rag_core/cli.py
```

## Folder Conventions

- `docs/` contains project guides and operational documentation.
- `data/source_docs/` contains ingestable source files (`.pdf`, `.md`).
- Folder-tag behavior and ignore rules for ingestion are documented in:
  - [`docs/ingestion/README.md`](docs/ingestion/README.md)
  - [`data/source_docs/README.md`](data/source_docs/README.md)

## Project Structure

```text
advanced-rag/
├── rag_core/
├── docs/
├── data/source_docs/
├── examples/
├── evaluation/
├── docker/
└── tests/
```

## License

MIT
