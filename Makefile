.PHONY: up down reset logs stat migrate-up migrate-down migrate-new migrate-status migrate-statusd ingest register-ingestion-deployment register-ingestion-deployments run-ingestion-flow test eval register-eval-deployment

ENV_FILE:= .env
COMPOSE_FILE:=docker/docker-compose.yaml
ALEMBIC_CONFIG:=db/alembic.ini
ALEMBIC_BASE:=$(if $(shell command -v uv 2>/dev/null),uv run alembic,$(if $(wildcard .venv/bin/alembic),.venv/bin/alembic,))
HOST_POSTGRES_HOST?=localhost
HOST_QDRANT_URL?=http://localhost:6333
HOST_PREFECT_API_URL?=http://localhost:4200/api
HOST_OTEL_EXPORTER_OTLP_TRACES_ENDPOINT?=http://localhost:6006/v1/traces
HOST_ENV:=POSTGRES_HOST=$(HOST_POSTGRES_HOST) QDRANT_URL=$(HOST_QDRANT_URL) PREFECT_API_URL=$(HOST_PREFECT_API_URL) OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=$(HOST_OTEL_EXPORTER_OTLP_TRACES_ENDPOINT)
ALEMBIC:=$(HOST_ENV) $(ALEMBIC_BASE) -c $(ALEMBIC_CONFIG)

# Container name
# 1. Capture the "rest" of the arguments
ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
# 2. Prevent 'make' from complaining that the container name isn't a valid command
$(eval $(ARGS):;@:)

up:
	docker compose --project-directory . --env-file $(ENV_FILE) -f $(COMPOSE_FILE) up -d --wait

down:
	docker compose --project-directory . --env-file $(ENV_FILE) -f $(COMPOSE_FILE) down

reset:
	@echo "🛑 Stopping and removing containers (including volumes)..."
	docker compose --project-directory . --env-file $(ENV_FILE) -f $(COMPOSE_FILE) down --volumes
	@echo "🧹 Sweeping the Docker floor (Removing dangling images and dead build cache)..."
	docker image prune -f
	docker builder prune -f
	@echo "🚀 Rebuilding and starting up..."
	docker compose --project-directory . --env-file $(ENV_FILE) -f $(COMPOSE_FILE) up --build -d --wait

logs:
	docker compose --project-directory . -f $(COMPOSE_FILE) logs -f --tail=100 $(ARGS)

stat:
	docker stats

migrate-up:
	@test -n "$(ALEMBIC_BASE)" || (echo "Alembic not found. Install uv or create .venv with alembic."; exit 1)
	$(ALEMBIC) upgrade $(if $(REV),$(REV),head)

migrate-down:
	@test -n "$(ALEMBIC_BASE)" || (echo "Alembic not found. Install uv or create .venv with alembic."; exit 1)
	$(ALEMBIC) downgrade $(if $(REV),$(REV),-1)

migrate-new:
	@test -n "$(ALEMBIC_BASE)" || (echo "Alembic not found. Install uv or create .venv with alembic."; exit 1)
	@test -n "$(MSG)" || (echo "MSG is required. Usage: make migrate-new MSG=\"your message\""; exit 1)
	$(ALEMBIC) revision --autogenerate -m "$(MSG)"

migrate-status:
	@test -n "$(ALEMBIC_BASE)" || (echo "Alembic not found. Install uv or create .venv with alembic."; exit 1)
	$(ALEMBIC) history --verbose

migrate-statusd: migrate-status

ingest: migrate-up
	@echo "Triggering Prefect Ingestion Flow inside Docker..."
	@if [ -n "$(MAX_JOBS)" ]; then \
		docker compose --project-directory . --env-file $(ENV_FILE) -f $(COMPOSE_FILE) run --rm --entrypoint "uv" ingestion-worker run --no-sync python -m orchestration.ingestion.flows end_to_end --max-jobs $(MAX_JOBS); \
	else \
		docker compose --project-directory . --env-file $(ENV_FILE) -f $(COMPOSE_FILE) run --rm --entrypoint "uv" ingestion-worker run --no-sync python -m orchestration.ingestion.flows end_to_end; \
	fi

test:
	uv run pytest tests/

eval:
	@echo "🔍 Checking worker environment for evaluation dependencies..."
	@docker compose --project-directory . --env-file $(ENV_FILE) -f $(COMPOSE_FILE) run --rm --entrypoint python ingestion-worker -c "import ragas" 2>/dev/null || \
		(echo "⚠️ Missing Eval dependencies! Auto-upgrading image (this will take a few minutes)..." && \
		 docker compose --project-directory . --env-file $(ENV_FILE) -f $(COMPOSE_FILE) build --build-arg UV_SYNC_EXTRAS="--extra ingestion --extra doc-processing --extra eval --extra rerank" ingestion-worker && \
		 docker compose --project-directory . --env-file $(ENV_FILE) -f $(COMPOSE_FILE) up -d --force-recreate --wait ingestion-worker && \
		 echo "✅ Eval dependencies installed and ingestion-worker restarted.")
	@echo "🚀 Environment verified. Running Evaluation..."
	docker compose --project-directory . --env-file $(ENV_FILE) -f $(COMPOSE_FILE) run --rm --entrypoint "uv" ingestion-worker run --no-sync python -c "from evaluation.orchestration.flows import eval_golden_flow; eval_golden_flow()"


register-eval-deployment:
	@echo "Registering eval deployments (requires EVAL_ENABLED=true)..."
	docker compose --project-directory . --env-file $(ENV_FILE) -f $(COMPOSE_FILE) run --rm --entrypoint "uv" ingestion-worker run --no-sync python -m orchestration.initialize
