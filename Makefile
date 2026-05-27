.PHONY: up down reset deep-clean logs stat migrate-up migrate-down migrate-new migrate-status migrate-statusd ingest register-ingestion-deployment register-ingestion-deployments run-ingestion-flow test eval register-eval-deployment

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
EVAL_RERANK_ENABLED?=true

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
	@echo "🛑 Stopping and removing containers..."
	docker compose --project-directory . --env-file $(ENV_FILE) -f $(COMPOSE_FILE) down --volumes
	@echo "🧹 Sweeping dangling ghost images..."
	docker image prune -f
	@echo "🚀 Rebuilding and starting up (using fast cache)..."
	docker compose --project-directory . --env-file $(ENV_FILE) -f $(COMPOSE_FILE) up --build -d --wait

deep-clean:
	@echo "🗑️ Clearing all Docker caches to reclaim hard drive space..."
	docker builder prune -a -f
	docker image prune -a -f
	@echo "✅ System is pristine. Next build will be from scratch."

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
	@echo "💡 Tip: You can also trigger this pipeline visually from the Prefect UI at http://localhost:4200 (Deployment: ingest-documents)"
	@if [ -n "$(MAX_JOBS)" ]; then \
		docker compose --project-directory . --env-file $(ENV_FILE) -f $(COMPOSE_FILE) run --rm --entrypoint "uv" ingestion-worker run --no-sync python -m orchestration.ingestion.flows end_to_end --max-jobs $(MAX_JOBS); \
	else \
		docker compose --project-directory . --env-file $(ENV_FILE) -f $(COMPOSE_FILE) run --rm --entrypoint "uv" ingestion-worker run --no-sync python -m orchestration.ingestion.flows end_to_end; \
	fi

test:
	uv run pytest tests/

eval:
	@echo "🔁 Building eval-capable worker image..."
	@docker compose --project-directory . --env-file $(ENV_FILE) -f $(COMPOSE_FILE) build --build-arg UV_SYNC_EXTRAS="--extra ingestion --extra doc-processing --extra eval" ingestion-worker
	@docker compose --project-directory . --env-file $(ENV_FILE) -f $(COMPOSE_FILE) up -d --force-recreate --wait ingestion-worker
	@echo "🔍 Checking worker environment for evaluation dependencies..."
	@docker compose --project-directory . --env-file $(ENV_FILE) -f $(COMPOSE_FILE) run --rm --entrypoint "uv" ingestion-worker run --no-sync python -c "from evaluation.runtime import assert_eval_runtime_ready; assert_eval_runtime_ready()"
	@echo "🚀 Environment verified. Running Evaluation..."
	@echo "💡 Tip: You can also trigger this pipeline visually from the Prefect UI at http://localhost:4200 (Deployment: evaluate-ground-truth)"
	docker compose --project-directory . --env-file $(ENV_FILE) -f $(COMPOSE_FILE) run --rm -e RERANK_ENABLED=$(EVAL_RERANK_ENABLED) --entrypoint "uv" ingestion-worker run --no-sync python -c "from evaluation.orchestration.flows import eval_golden_flow; eval_golden_flow()"


register-eval-deployment:
	@echo "Registering all Prefect deployments, including eval deployments..."
	docker compose --project-directory . --env-file $(ENV_FILE) -f $(COMPOSE_FILE) run --rm --entrypoint "uv" ingestion-worker run --no-sync python -m orchestration.initialize
