FROM python:3.11-slim AS base

WORKDIR /app

# Install uv for fast dependency management
RUN pip install --no-cache-dir uv

# Copy dependency files first (for layer caching)
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy application code
COPY rag_core/ rag_core/
COPY evaluation/ evaluation/

# Default command (override in docker-compose)
CMD ["uv", "run", "python", "-m", "rag_core.pipelines.pipeline"]
