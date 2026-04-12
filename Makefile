.PHONY: up down reset logs stat

ENV_FILE:= .env
COMPOSE_FILE:=docker/docker-compose.yaml
# Contaienr name
# 1. Capture the "rest" of the arguments
ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
# 2. Prevent 'make' from complaining that the container name isn't a valid command
$(eval $(ARGS):;@:)

up:
	docker compose --env-file $(ENV_FILE) -f $(COMPOSE_FILE) up -d

down:
	docker compose --env-file $(ENV_FILE) -f $(COMPOSE_FILE) down

reset:
	docker compose --env-file $(ENV_FILE) -f $(COMPOSE_FILE) down --volumes
	docker compose --env-file $(ENV_FILE) -f $(COMPOSE_FILE) up --build -d
logs:
	docker compose -f $(COMPOSE_FILE) logs -f --tail=100 $(ARG)

stat:
	docker stats
