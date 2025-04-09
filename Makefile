SHELL=/bin/bash
APP_NAME?=langgate

COMPOSE_PROJECT?=${APP_NAME}
COMPOSE_PROFILES?=s,e

LOG_LEVEL?=info

sync:
	@uv sync --all-extras --dev --all-packages

run-local-no-proxy: sync
	@-COMPOSE_PROFILES=a docker compose down
	LOG_LEVEL=${LOG_LEVEL} \
	uv run uvicorn langgate.server.main:app \
	--host 0.0.0.0 --port 4000 \
	--reload --log-level ${LOG_LEVEL}

# For local development only - detect host IP when Docker networking fails
# This is a workaround and should not be used in production
HOST_IP?=$(shell ./docker/scripts/detect_local_dev_ip.sh)

# Run with Envoy proxy in Docker and Python server on host
run-local: sync
	@-COMPOSE_PROFILES=a docker compose down
	@echo "Host IP detected as: ${HOST_IP}"
	@echo "Starting Envoy proxy in Docker connecting to host machine..."
	@HOST_IP=${HOST_IP} COMPOSE_PROFILES=e docker compose -f docker-compose.yaml \
	-f docker/compose/docker-compose.local.host.yaml up --build -d
	@echo "Starting Python server on host machine (0.0.0.0:4000)..."
	@echo "Envoy proxy will connect to host using IP: ${HOST_IP}"
	@echo "You can access the API docs at http://localhost:10000/api/v1/docs"
	LOG_LEVEL=${LOG_LEVEL} \
	uv run uvicorn langgate.server.main:app \
	--host 0.0.0.0 --port 4000 \
	--reload --log-level ${LOG_LEVEL}

mypy:
	@echo "Running mypy on each package..."
	@cd packages/sdk && uv run mypy --config-file=../../pyproject.toml --package=langgate.sdk
	@cd packages/core && uv run mypy --config-file=../../pyproject.toml --package=langgate.core
	@cd packages/client && uv run mypy --config-file=../../pyproject.toml --package=langgate.client
	@cd packages/processor && uv run mypy --config-file=../../pyproject.toml --package=langgate.processor
	@cd packages/registry && uv run mypy --config-file=../../pyproject.toml --package=langgate.registry
	@cd packages/server && uv run mypy --config-file=../../pyproject.toml --package=langgate.server
	@cd packages/transform && uv run mypy --config-file=../../pyproject.toml --package=langgate.transform

ruff:
	@uv run ruff format .
	@uv run ruff check --fix

pre-commit-install:
	@uv run pre-commit install

pre-commit-uninstall:
	@uv run pre-commit uninstall

pre-commit-run:
	@uv run pre-commit run --all-files

hadolint:
	@find . -type f \( -iname '*Dockerfile*' -or -iname '*.dockerfile' \) -print0 | xargs -0 hadolint

lint: ruff mypy pre-commit-run hadolint

test: sync
	@uv run pytest tests \
	-c pyproject.toml --cov-config=setup.cfg \
	-n auto -rpF

test-flake-finder: sync
	@uv run pytest tests \
	-c pyproject.toml --cov-config=setup.cfg \
	-n auto -rpF --flake-finder

gen-coverage-badge:
	@uv run genbadge coverage \
	-i coverage_report/coverage.xml \
	-o coverage_report/coverage-badge.svg

gen-test-count-badge:
	@test_count=$$(uv run pytest --collect-only | tail -n 1 | grep -o '[0-9]\+ tests' | grep -o '[0-9]\+'); \
	echo '<?xml version="1.0"?><testcount>'$$test_count'</testcount>' > coverage_report/test-count.xml

COMPOSE_TIMEOUT?=60
compose-healthcheck:
	@timeout=${COMPOSE_TIMEOUT}; \
	while [ "$$timeout" -gt 0 ] && [ "$$(docker compose -p $(COMPOSE_PROJECT) ps | grep -E 'starting|unhealthy')" ]; do \
		echo "Waiting for services to become healthy..."; \
		sleep 5; \
		timeout=$$((timeout - 5)); \
	done; \
	if [ "$$timeout" -le 0 ]; then \
		echo "Timed out waiting for services to become healthy"; \
		exit 1; \
	fi; \
	if [ "$$(docker compose -p $(COMPOSE_PROJECT) ps | grep 'unhealthy')" ]; then \
		echo "One of more services are unhealthy"; \
		exit 1; \
	fi; \
	echo "All services are healthy"


BUILD?=1
compose-up:
	@cmd="docker compose --env-file .env up"; \
	if [ "${BUILD}" -eq 1 ]; then \
		cmd="$$cmd --build"; \
	fi; \
	LOG_LEVEL=${LOG_LEVEL} \
	COMPOSE_BAKE=true \
	COMPOSE_PROFILES=${COMPOSE_PROFILES} \
	$$cmd -d; \
	${MAKE} compose-healthcheck

# Development mode with hot reloading using Docker Compose watch
compose-dev:
	@cmd="docker compose --env-file .env -f docker-compose.yaml -f docker/compose/docker-compose.dev.yaml"; \
	if [ "${BUILD}" -eq 1 ]; then \
		$$cmd build; \
	fi; \
	LOG_LEVEL=${LOG_LEVEL} \
	COMPOSE_BAKE=true \
	COMPOSE_PROFILES=${COMPOSE_PROFILES} \
	$$cmd up --watch; \
	${MAKE} compose-healthcheck


compose-down:
	@COMPOSE_PROFILES=${COMPOSE_PROFILES} docker compose down

compose-breakdown:
	@COMPOSE_PROFILES=a docker compose down -v

COMPOSE_PROFILES_TEST?=e
compose-test-containers:
	@cmd="docker compose -f tests/docker-compose.yaml --env-file .env up"; \
	if [ "${BUILD}" -eq 1 ]; then \
		cmd="$$cmd --build"; \
	fi; \
	COMPOSE_PROFILES=${COMPOSE_PROFILES_TEST} $$cmd -d;
	${MAKE} compose-healthcheck COMPOSE_PROJECT=tests

compose-down-test-containers:
	@COMPOSE_PROFILES=a \
	docker compose -f tests/docker-compose.yaml --env-file .env down

compose-breakdown-test-containers:
	@COMPOSE_PROFILES=a \
	docker compose -f tests/docker-compose.yaml --env-file .env down -v

update-model-costs:
	@uv run update_costs

uv-build:
	@uv build --force-pep517 --all-packages

uv-publish: uv-build
	@uv publish --trusted-publishing automatic --no-cache dist/*

PYINDEX_HOST?=localhost
PYINDEX_PORT?=8081
uv-publish-to-local-index: uv-build
	@uv publish \
		--publish-url http://${PYINDEX_HOST}:${PYINDEX_PORT} \
		--check-url http://${PYINDEX_HOST}:${PYINDEX_PORT}/simple/ \
		-u ${PYINDEX_USER} \
		-p ${PYINDEX_PASS} \
		--no-cache \
		dist/*


helm-deps:
	@$(MAKE) -C deployment/k8s helm-deps

helm-lint:
	@$(MAKE) -C deployment/k8s helm-lint


define bump-version-and-update
	@echo "Bumping version $(if $(2),to $(2),$(1))..."
	@./scripts/bump_version.py $(if $(2),--version $(2),$(1))
	@${MAKE} validate-versions
	@${MAKE} sync
	@${MAKE} helm-deps
	@${MAKE} helm-lint
endef

bump-patch:
	$(call bump-version-and-update,patch)

bump-minor:
	$(call bump-version-and-update,minor)

bump-major:
	$(call bump-version-and-update,major)

bump-version:
	@if [ -z "$(VERSION)" ]; then \
		echo "Usage: make bump-version VERSION=X.Y.Z"; \
		exit 1; \
	fi
	$(call bump-version-and-update,,$(VERSION))

validate-versions:
	@echo "Validating version consistency..."
	@./scripts/bump_version.py --validate
