.PHONY: up-core up down reset reload deep-clean logs stat cli ui migrate-up migrate-down migrate-new migrate-status migrate-statusd ingest test eval eval-drift eval-synthetic register-eval-deployment

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

up-core:
	@echo "🚀 Starting backend infrastructure (DB, Qdrant, Phoenix, Prefect, Worker, API)..."
	docker compose --project-directory . --env-file $(ENV_FILE) -f $(COMPOSE_FILE) up -d --wait rag_api rag-worker prefect-server phoenix

up: up-core
	@echo "🖥️  Starting Open-WebUI..."
	docker compose --project-directory . --env-file $(ENV_FILE) -f $(COMPOSE_FILE) up -d --wait web_ui

down:
	docker compose --project-directory . --env-file $(ENV_FILE) -f $(COMPOSE_FILE) down

reset:
	@echo "🛑 Stopping and removing containers..."
	docker compose --project-directory . --env-file $(ENV_FILE) -f $(COMPOSE_FILE) down --volumes
	@echo "🧹 Sweeping dangling ghost images..."
	docker image prune -f
	@echo "🚀 Rebuilding and starting up (using fast cache)..."
	docker compose --project-directory . --env-file $(ENV_FILE) -f $(COMPOSE_FILE) up --build -d --wait

reload:
	@echo "♻️ Recreating containers to apply new .env values..."
	docker compose --project-directory . --env-file $(ENV_FILE) -f $(COMPOSE_FILE) up -d --force-recreate

deep-clean:
	@echo "🗑️ Clearing all Docker caches to reclaim hard drive space..."
	docker builder prune -a -f
	docker image prune -a -f
	@echo "✅ System is pristine. Next build will be from scratch."

logs:
	docker compose --project-directory . -f $(COMPOSE_FILE) logs -f --tail=100 $(ARGS)

stat:
	docker stats

cli: up-core
	@echo "🚀 Launching interactive RAG CLI..."
	docker compose --project-directory . --env-file $(ENV_FILE) -f $(COMPOSE_FILE) run --rm rag_cli

ui: up-core
	@echo "🚀 Launching local UI using RAG_LOCAL_UI_PORT from $(ENV_FILE) (default: 8501)..."
	@echo "Note: Streamlit mode uses a dev-only examples/ mount for hot reloading."
	docker compose --project-directory . --env-file $(ENV_FILE) -f $(COMPOSE_FILE) run --rm --service-ports rag_ui

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

ingest: up-core migrate-up
	@echo "Triggering Prefect Ingestion Flow inside Docker..."
	@printf "\033[36m💡 Tip: You can also trigger this pipeline visually from the Prefect UI at http://localhost:4200 (Deployment: ingest-documents)\033[0m\n"
	@if [ -n "$(MAX_JOBS)" ]; then \
		docker compose --project-directory . --env-file $(ENV_FILE) -f $(COMPOSE_FILE) exec rag-worker .venv/bin/python -m orchestration.ingestion.flows end_to_end --max-jobs $(MAX_JOBS); \
	else \
		docker compose --project-directory . --env-file $(ENV_FILE) -f $(COMPOSE_FILE) exec rag-worker .venv/bin/python -m orchestration.ingestion.flows end_to_end; \
	fi

test:
	uv run pytest tests/

eval: up-core
	@echo "🚀 Running Ground Truth Evaluation..."
	@printf "\033[36m💡 Tip: You can also trigger this pipeline visually from the Prefect UI at http://localhost:4200 (Deployment: evaluate-ground-truth)\033[0m\n"
	docker compose --project-directory . --env-file $(ENV_FILE) -f $(COMPOSE_FILE) exec -e RERANK_ENABLED=$(EVAL_RERANK_ENABLED) rag-worker .venv/bin/python -m orchestration.evaluation.run_eval

eval-drift: up-core
	@echo "🚀 Running Drift Evaluation..."
	@printf "\033[36m💡 Tip: You can also trigger this pipeline visually from the Prefect UI at http://localhost:4200 (Deployment: evaluate-traces)\033[0m\n"
	docker compose --project-directory . --env-file $(ENV_FILE) -f $(COMPOSE_FILE) exec -e RERANK_ENABLED=$(EVAL_RERANK_ENABLED) rag-worker .venv/bin/python -m orchestration.evaluation.run_drift

eval-synthetic: up-core
	@echo "🚀 Generating Synthetic QA Dataset..."
	@printf "\033[36m💡 Tip: You can also trigger this pipeline visually from the Prefect UI at http://localhost:4200 (Deployment: generate-synthetic-qa)\033[0m\n"
	docker compose --project-directory . --env-file $(ENV_FILE) -f $(COMPOSE_FILE) exec rag-worker .venv/bin/python -m orchestration.evaluation.run_synthetic

register-eval-deployment: up-core
	@echo "Registering all Prefect deployments, including eval deployments..."
	docker compose --project-directory . --env-file $(ENV_FILE) -f $(COMPOSE_FILE) exec rag-worker .venv/bin/python -m orchestration.initialize
