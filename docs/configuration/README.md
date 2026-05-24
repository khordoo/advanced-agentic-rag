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

## Retrieval RRF Tuning

For backend-specific hybrid retrieval semantics, RRF constants, and why
Postgres defaults to `k=2`, see:
- [Retrieval RRF Semantics](retrieval-rrf.md)
  Note: optional integer env vars must be omitted/commented out when unused; empty values are invalid.

## Hugging Face & Model Prefetching

The project prefetches embedding and reranker models during Docker builds to
minimize startup latency.

- `HF_TOKEN` (Optional): Only required if you change the default models to a
  gated model (e.g., Llama-3) or if your build environment is hitting Hugging
  Face rate limits. For standard public models, leave this blank.

## Adding an Embedding Provider

`EMBEDDING_PROVIDER` is intentionally validated against the built-in
`EmbeddingProvider` enum. This catches typos early; adding a new provider is a
code change, not just a `.env` change.

To add a provider:

- Add its name to `EmbeddingProvider` in `rag_core.embeddings.types`.
- Implement a `BaseEmbeddingModel` adapter for document/query embeddings.
- Register the adapter in `create_embedding_model()`.
- Add provider-specific settings only if the adapter needs them.
- Add tests for settings validation, factory behavior, dimensions, and failure
  modes.

Currently supported provider values are `fastembed` and `openai`. `openai` is
reserved as a placeholder until its adapter is implemented.
