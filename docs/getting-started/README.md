# Getting Started

Use this guide for first-time setup and initial validation.

## Setup

```bash
cp .env.example .env
uv sync --extra dev
```

## Run

```bash
# Ingestion pipeline
uv run rag_core/pipelines/ingestion.py

# Interactive CLI
uv run rag_core/cli.py
```

## Next

- Continue with [Configuration](../configuration/README.md).
- Learn ingestion rules in [Ingestion](../ingestion/README.md).
