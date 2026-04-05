# 📚 Learning Examples

Interactive scripts that demonstrate how the Advanced RAG pipeline works,
step by step. **No infrastructure required** — everything runs in-memory.

## Prerequisites

```bash
# From the project root
uv sync
```

## Scripts

| # | Script | What it Demonstrates |
|---|---|---|
| 01 | `01_chunking_basics.py` | Markdown header splitting + recursive character splitting |
| 02 | `02_parent_child.py` | Parent-child hierarchical retrieval pattern |
| 03 | `03_reranking.py` | How cross-encoder reranking improves result quality |
| 04 | `04_full_pipeline.py` | End-to-end: ingest PDF → query → get answer |

## Running

```bash
# Run any script individually
python examples/01_chunking_basics.py

# For the full pipeline (requires a .env with LLM provider config)
python examples/04_full_pipeline.py
```

## Sample Documents

The `sample_docs/` folder contains a small demo PDF for the examples.
