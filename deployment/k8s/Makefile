# Makefile for LangGate Kubernetes resources
SCRIPTS_DIR = ./scripts

helm-deps:
	$(SCRIPTS_DIR)/chart-dependencies.sh

helm-lint: helm-deps
	$(SCRIPTS_DIR)/chart-lint.sh

helm-test: helm-deps
	$(SCRIPTS_DIR)/chart-test.sh

helm-test-config: helm-deps
	$(SCRIPTS_DIR)/chart-test-with-config.sh

helm-unittest: helm-deps
	$(SCRIPTS_DIR)/chart-unittest.sh
