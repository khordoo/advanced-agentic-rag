# Retrieval RRF Semantics

This project supports two retrieval backends with different hybrid-retrieval behavior:

- `qdrant`: native backend fusion in Qdrant
- `postgres`: application-level weighted RRF over dense + FTS candidates

## Why Scores Differ Across Backends

Do not expect strict score or rank parity between backends:

- Qdrant sparse retrieval and Postgres FTS are different retrieval signals.
- Fusion/scoring is executed in different layers (database-native vs app-layer).
- Even with the same query and filters, ordering can differ.

This is expected and not a bug.

## RRF Controls

Runtime controls (from `.env`):

- `RETRIEVAL_BACKEND=qdrant|postgres`
- `RETRIEVAL_HYBRID_ALPHA=0.0..1.0`
- `RETRIEVAL_POSTGRES_RRF_K` (app-level Postgres weighted-RRF constant)
- `RETRIEVAL_QDRANT_RRF_K` (optional explicit Qdrant native RRF constant)

### Optional Integer Env Note

`RETRIEVAL_QDRANT_RRF_K` is optional, but when provided it must be a valid integer.

- Valid: `RETRIEVAL_QDRANT_RRF_K=60`
- Default behavior: comment it out / omit it
- Invalid: `RETRIEVAL_QDRANT_RRF_K=` (empty value; fails settings parsing)

### `hybrid_alpha`

`hybrid_alpha` is the single retrieval balance dial:

- dense weight = `hybrid_alpha`
- sparse/text weight = `1 - hybrid_alpha`

## Why Postgres Defaults to `k=2`

`RETRIEVAL_POSTGRES_RRF_K=2` is intentional for RAG head precision.

With smaller `k`, rank-1 / rank-2 hits get much stronger boost than mid-tail consensus.
For context construction in RAG, this usually improves precision of top context chunks.

Tradeoff:

- smaller `k` (`2`): sharper top-hit bias, less smoothing
- larger `k` (`30-60`): smoother consensus behavior, more robust to single-ranker noise

If results look overly spiky, increase `RETRIEVAL_POSTGRES_RRF_K`.
If exact high-confidence hits are buried, decrease it.

## Qdrant Default Behavior

By default, Qdrant uses native RRF defaults unless you explicitly set:

- `RETRIEVAL_QDRANT_RRF_K`

This keeps existing Qdrant behavior stable and avoids surprise production changes.
