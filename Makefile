.PHONY: up-core up down reset reload deep-clean logs stat cli ui migrate-up migrate-down migrate-new migrate-status migrate-statusd ingest test test-api test-worker test-all eval eval-drift eval-synthetic register-eval-deployment

ENV_FILE:= .env
COMPOSE_FILE:=docker/docker-compose.yaml
ALEMBIC_CONFIG:=db/alembic.ini
COMPOSE_RUN_API:=docker compose --project-directory . --env-file $(ENV_FILE) -f $(COMPOSE_FILE) run --rm --no-deps rag_api
ALEMBIC:=$(COMPOSE_RUN_API) /app/.venv/bin/alembic -c $(ALEMBIC_CONFIG)
EVAL_STATIC_RERANK_ENABLED ?= true
PYTEST := uv run --no-dev --with pytest --with pytest-asyncio pytest

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
	@printf "\033[31m⚠️  WARNING: reset will delete all Docker volumes, cached images, and local container state.\033[0m\n"
	@printf "\033[31mAre you sure? Type reset to continue: \033[0m"; \
	read -r confirm; \
	if [ "$$confirm" != "reset" ]; then \
		echo "Aborted."; \
		exit 1; \
	fi
	docker compose --project-directory . --env-file $(ENV_FILE) -f $(COMPOSE_FILE) down --volumes
	@echo "🧹 Sweeping dangling ghost images..."
	docker image prune -f
	@echo "🚀 Rebuilding and starting up (using fast cache)..."
	docker compose --project-directory . --env-file $(ENV_FILE) -f $(COMPOSE_FILE) up --build -d --wait

reload:
	@echo "♻️ Restarting containers to apply new .env values..."
	docker compose --project-directory . --env-file $(ENV_FILE) -f $(COMPOSE_FILE) down
	docker compose --project-directory . --env-file $(ENV_FILE) -f $(COMPOSE_FILE) up -d --wait

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
	@echo "🗄️ Running Alembic migrations inside Docker..."
	$(ALEMBIC) upgrade $(if $(REV),$(REV),head)

migrate-down:
	@echo "🗄️ Rolling back Alembic migrations inside Docker..."
	$(ALEMBIC) downgrade $(if $(REV),$(REV),-1)

migrate-new:
	@test -n "$(MSG)" || (echo "MSG is required. Usage: make migrate-new MSG=\"your message\""; exit 1)
	@echo "🗄️ Creating Alembic migration inside Docker..."
	$(ALEMBIC) revision --autogenerate -m "$(MSG)"

migrate-status:
	@echo "🗄️ Showing Alembic migration history inside Docker..."
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
	@echo "⚡ Running fast local tests..."
	$(PYTEST) tests/test_*.py

test-api: up-core
	@echo "🧪 Ensuring infra is up, then running API tests in the live api container..."
	docker compose --project-directory . --env-file $(ENV_FILE) -f $(COMPOSE_FILE) exec rag_api $(PYTEST) tests/api

test-worker: up-core
	@echo "🧪 Ensuring infra is up, then running worker tests in the live rag-worker container..."
	docker compose --project-directory . --env-file $(ENV_FILE) -f $(COMPOSE_FILE) exec rag-worker /bin/sh -lc '$(PYTEST) tests/worker'

test-all: test test-api test-worker
	@echo "✅ All test suites passed!"

eval: up-core
	@echo "🚀 Running Ground Truth Evaluation..."
	@printf "\033[36m💡 Tip: You can also trigger this pipeline visually from the Prefect UI at http://localhost:4200 (Deployment: evaluate-ground-truth)\033[0m\n"
	@if [ -n "$(DATASET)" ]; then \
		_name=$${NAME:-$$(basename "$(DATASET)" .csv | tr '_' '-')}; \
		printf "\033[33m📂 Dataset: $(DATASET) → Phoenix name: $${_name}\033[0m\n"; \
		docker compose --project-directory . --env-file $(ENV_FILE) -f $(COMPOSE_FILE) \
			exec -e RERANK_ENABLED=$(EVAL_STATIC_RERANK_ENABLED) \
			-e EVAL_STATIC_DATASET_CSV_PATH=$(DATASET) \
			-e EVAL_STATIC_DATASET_NAME=$${_name} \
			rag-worker .venv/bin/python -m orchestration.evaluation.run_eval; \
	else \
		docker compose --project-directory . --env-file $(ENV_FILE) -f $(COMPOSE_FILE) \
			exec -e RERANK_ENABLED=$(EVAL_STATIC_RERANK_ENABLED) \
			rag-worker .venv/bin/python -m orchestration.evaluation.run_eval; \
	fi

eval-drift: up-core
	@echo "🚀 Running Drift Evaluation..."
	@printf "\033[36m💡 Tip: You can also trigger this pipeline visually from the Prefect UI at http://localhost:4200 (Deployment: evaluate-traces)\033[0m\n"
	docker compose --project-directory . --env-file $(ENV_FILE) -f $(COMPOSE_FILE) exec -e RERANK_ENABLED=$(EVAL_STATIC_RERANK_ENABLED) rag-worker .venv/bin/python -m orchestration.evaluation.run_drift

eval-synthetic: up-core
	@echo "🚀 Generating Synthetic QA Dataset..."
	@printf "\033[36m💡 Tip: You can also trigger this pipeline visually from the Prefect UI at http://localhost:4200 (Deployment: generate-synthetic-qa)\033[0m\n"
	docker compose --project-directory . --env-file $(ENV_FILE) -f $(COMPOSE_FILE) exec rag-worker .venv/bin/python -m orchestration.evaluation.run_synthetic

register-eval-deployment: up-core
	@echo "Registering all Prefect deployments, including eval deployments..."
	docker compose --project-directory . --env-file $(ENV_FILE) -f $(COMPOSE_FILE) exec rag-worker .venv/bin/python -m orchestration.initialize
