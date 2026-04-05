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
                                  ▼
                     ┌─────────────────────────┐
                     │  Markdown Header Split   │
                     │  (h1 > h2 > h3 sections) │
                     └────────────┬────────────┘
                                  │
                     ┌────────────┴────────────┐
                     ▼                         ▼
              ┌─────────────┐          ┌─────────────┐
              │   Parent     │          │   Child      │
              │  (~1500 ch)  │          │  (~300 ch)   │
              │  Doc Store   │          │  Vector DB   │
              └──────┬──────┘          └──────┬──────┘
                     │                        │
                     │    ┌───────────────────┘
                     │    │  similarity search
                     │    ▼
                     │  ┌─────────────────────┐
                     │  │  Cross-Encoder       │
                     │  │  Reranking           │
                     │  └─────────┬───────────┘
                     │            │
                     │◀───────────┘  parent lookup
                     │
                     ▼
              ┌─────────────┐
              │     LLM      │
              │  Generation   │
              └─────────────┘
```

## Features

- **Hierarchical Chunking** — Markdown header splitting → parent/child chunks with page-level metadata
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
python examples/01_chunking_basics.py

# See parent-child retrieval in action
python examples/02_parent_child.py

# See reranking improvement
python examples/03_reranking.py
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
