# Configuration

Project runtime behavior is configured through `.env` (see `.env.example`).

## Key Configuration Areas

- Vector/indexing and storage paths.
- Retrieval and chunking parameters.
- Generation provider, model IDs, and guardrail limits.
- Prompt profile overrides (`PROMPTS_PROFILE_PATH`, `PROMPTS_PROFILE_STRICT`).

## Important Paths

- `DOCS_DIR` defaults to `data/source_documents`.
- `PROCESSED_DOCS_DIR` defaults to `data/processed`.

## Generation Models

For reasoning/verifier/fast model setup and tuning, see:
- [Generation Model Settings](generation-models.md)
