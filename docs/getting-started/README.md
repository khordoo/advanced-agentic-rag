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
uv run python -m rag_core.ingestion.pipeline

# Interactive CLI
uv run rag_core/cli.py
```

## Updating Configuration (DX)

When you modify environment variables in your `.env` file (e.g., switching from FastEmbed to Bedrock), Docker Compose does not automatically pick up these changes if you simply restart the container. The container must be recreated to inject the new variables.

To apply configuration changes efficiently without losing database volumes, use:

```bash
make reload
```

### Rule of Thumb
- **Runtime changes** (e.g., changing an API key, switching a provider, updating weights): Use `make reload`.
- **Build-time changes** (e.g., adding a new Python package, changing the Dockerfile): Use `make reset`.

## Next

- Continue with [Configuration](../configuration/README.md).
- Learn ingestion rules in [Ingestion](../ingestion/README.md).
- Use the API guide in [OpenAI-Compatible API](openai-api.md).
