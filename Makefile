.PHONY: up down reset logs stat migrate-up migrate-down migrate-new migrate-status migrate-statusd register-ingestion-deployment register-ingestion-deployments run-ingestion-flow

ENV_FILE:= .env
COMPOSE_FILE:=docker/docker-compose.yaml
ALEMBIC_CONFIG:=db/alembic.ini
ALEMBIC_BASE:=$(if $(shell command -v uv 2>/dev/null),uv run alembic,$(if $(wildcard .venv/bin/alembic),.venv/bin/alembic,))
ALEMBIC:=$(ALEMBIC_BASE) -c $(ALEMBIC_CONFIG)
# Contaienr name
# 1. Capture the "rest" of the arguments
ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
# 2. Prevent 'make' from complaining that the container name isn't a valid command
$(eval $(ARGS):;@:)

up:
	docker compose --project-directory . --env-file $(ENV_FILE) -f $(COMPOSE_FILE) up -d

down:
	docker compose --project-directory . --env-file $(ENV_FILE) -f $(COMPOSE_FILE) down

reset:
	docker compose --project-directory . --env-file $(ENV_FILE) -f $(COMPOSE_FILE) down --volumes
	docker compose --project-directory . --env-file $(ENV_FILE) -f $(COMPOSE_FILE) up --build -d
logs:
	docker compose --project-directory . -f $(COMPOSE_FILE) logs -f --tail=100 $(ARG)

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
	@echo "Ensuring all local dependencies are synced..."
	uv sync --all-extras
	@echo "Starting Ingestion Pipeline..."
	uv run python -m rag_core.ingestion.pipeline

register-ingestion-deployment:
	uv run --extra ingestion python -m orchestration.ingestion.flows end_to_end \
		--register-deployment \
		--deployment-name $${DEPLOYMENT_NAME:-local-ingestion-runner} \
		--work-pool $${WORK_POOL:-ingestion-pool}

register-ingestion-deployments:
	uv run --extra ingestion python -m orchestration.ingestion.flows end_to_end \
		--register-all-deployments \
		--deployment-name $${DEPLOYMENT_NAME:-local-ingestion-runner} \
		--work-pool $${WORK_POOL:-ingestion-pool}

run-ingestion-flow:
	@if [ -n "$(MAX_JOBS)" ]; then \
		uv run --extra ingestion python -m orchestration.ingestion.flows end_to_end --max-jobs $(MAX_JOBS); \
	else \
		uv run --extra ingestion python -m orchestration.ingestion.flows end_to_end; \
	fi
