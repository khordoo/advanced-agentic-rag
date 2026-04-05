# 🐳 Docker Production Stack

## Quick Start

```bash
# Start Qdrant + LangFuse
docker compose up -d

# Check services are healthy
docker compose ps

# View LangFuse UI
open http://localhost:3000

# Qdrant dashboard
open http://localhost:6333/dashboard
```

## Services

| Service | Port | Purpose |
|---|---|---|
| **Qdrant** | `6333` (REST), `6334` (gRPC) | Vector DB with hybrid search (BM25 + vector) |
| **LangFuse** | `3000` | LLM observability, tracing, and evaluation |
| **PostgreSQL** | internal | LangFuse backend database |

## Phase 3 Services (Coming Soon)

| Service | Port | Purpose |
|---|---|---|
| **app** (FastAPI) | `8000` | RAG API with OpenAI-compatible endpoint |
| **Chainlit** | `8501` | Chat UI with source document display |

## Data Persistence

All data is stored in Docker volumes:
- `qdrant_data` — Vector DB storage
- `langfuse_pg_data` — LangFuse PostgreSQL data

To reset:
```bash
docker compose down -v  # -v removes volumes
```

## Connecting Your App

While the FastAPI container isn't ready yet, you can connect your local
Python scripts to the Docker services:

```python
from rag_core.config import load_config
import os

os.environ["QDRANT_URL"] = "http://localhost:6333"
os.environ["LANGFUSE_HOST"] = "http://localhost:3000"

config = load_config()
```
