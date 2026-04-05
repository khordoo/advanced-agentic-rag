# Advanced RAG Pipeline

A production-grade Retrieval-Augmented Generation pipeline featuring hierarchical parent-child chunking, cross-encoder reranking, hybrid search (BM25 + vector), and automated evaluation with RAGAS.

## Architecture

```
PDF Document
    │
    ▼
┌──────────────┐     ┌─────────────────────────┐
│   Docling     │────▶│  Numbered Markdown       │
│  (converter)  │     │  with page markers       │
└──────────────┘     └────────────┬────────────┘
                                  │
                         Stage 1: Split by headers
                         (preserve document structure)
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │  Markdown Sections       │
                     │  h1 > h2 > h3            │
                     │  (can be very long)       │
                     └────────────┬────────────┘
                                  │
                         Stage 2: Recursive split
                         (sections → manageable parents)
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │  Parent Chunks (~1500ch)  │
                     │  → stored in Doc Store    │
                     └────────────┬────────────┘
                                  │
                         Stage 3: Sub-split
                         (parents → small children)
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │  Child Chunks (~300ch)    │
                     │  → embedded in Vector DB  │
                     └────────────┬────────────┘
                                  │
                         Query time: search children
                         → rerank → return parents
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │  Cross-Encoder Reranking  │
                     │  → deduplicate parents    │
                     │  → LLM generation         │
                     └─────────────────────────┘
```

## How the Chunking Works (and Why)

The pipeline uses a **3-stage chunking strategy**. Each stage exists for a specific reason:

**Stage 1 — Markdown Header Split.**
We first split the document along its natural heading boundaries (`#`, `##`, `###`). This preserves the document's logical structure — a section about "Risk Factors" stays together, with its header metadata attached. Without this step, a naive character splitter would cut right through the middle of a section, losing the context of *which part of the document* a chunk belongs to.

**Stage 2 — Recursive Split into Parents.**
But a single section can be enormous (thousands of characters). A section like "Financial Highlights" might span 3 pages. So we use a recursive character splitter to break each section into manageable **parent chunks** (~1,500 chars). These parents inherit the section metadata (headers, page numbers) and are stored in a document store. This is the text the LLM will actually read — large enough to contain real context.

**Stage 3 — Sub-split into Children.**
For vector search, smaller is better — shorter text produces more focused embeddings that match queries more precisely. So each parent is further split into **child chunks** (~300 chars) that get embedded in the vector database. Each child stores a pointer back to its parent.

**At query time**, we search the small children (precision), rerank them with a cross-encoder, then look up and deduplicate the parents (context). The LLM receives the larger parent chunks so it has enough information to give a thorough answer.

## Features

- **3-Stage Hierarchical Chunking** — Header split → parent chunks → child chunks, with page-level metadata
- **Parent-Child Retrieval** — Search small chunks (precision), return large chunks (context)
- **Cross-Encoder Reranking** — BAAI/bge-reranker-v2-m3 for accurate re-scoring
- **Multi-Provider LLM** — AWS Bedrock, Ollama, LM Studio, Kimi (NVIDIA)
- **RAGAS Evaluation** — Faithfulness, relevancy, recall, precision, correctness
- **Synthetic QA Generation** — Auto-generate test sets from your documents
- **Docker Stack** — Qdrant (hybrid search) + LangFuse (observability)

## Quick Start

### 1. Install

```bash
# Clone and install
git clone <your-repo-url> && cd advanced-rag
uv sync

# For development
uv sync --extra dev

# For production (Qdrant)
uv sync --extra qdrant
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your provider credentials
```

### 3. Run Examples (No Infra Needed)

```bash
# See how chunking works
uv run python examples/01_chunking_basics.py

# See parent-child retrieval in action
uv run python examples/02_parent_child.py

# See reranking improvement
uv run python examples/03_reranking.py

# Interactive visualizer (requires: uv sync --extra demo)
uv run streamlit run examples/streamlit_app.py
```

### 4. Production Stack (Docker)

```bash
cd docker
docker compose up -d

# Qdrant dashboard: http://localhost:6333/dashboard
# LangFuse UI:      http://localhost:3000
```

## Project Structure

```
advanced-rag/
├── rag_core/           # Shared library (pip-installable)
│   ├── converter.py    # PDF → Markdown
│   ├── chunker.py      # Markdown → parent/child chunks
│   ├── retriever.py    # Hierarchical search + reranking
│   ├── pipeline.py     # End-to-end RAG pipeline
│   ├── agents.py       # LLM factory + prompt loader
│   └── prompts/        # YAML prompt templates
├── examples/           # Learning track (zero infra)
├── evaluation/         # RAGAS evaluation + QA generation
├── docker/             # Production stack (Qdrant + LangFuse)
├── tests/              # Test suite
└── docs/               # Source PDFs for ingestion
```

## Evaluation Results

With cross-encoder reranking on child chunks before parent lookup:

| Metric | Score |
|---|---|
| Faithfulness | 0.972 |
| Answer Relevancy | 0.903 |
| Context Recall | 0.917 |
| Context Precision | 0.917 |
| Answer Correctness | 0.836 |

## Roadmap

- [x] Phase 1: Restructure + Docker Compose (Qdrant + LangFuse)
- [ ] Phase 2: Streamlit frontend for learning examples
- [ ] Phase 3: FastAPI API layer + Chainlit chat UI
- [ ] Phase 4: Showcase test suite (pytest)

## License

MIT
