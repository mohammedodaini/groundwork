.PHONY: help install run test lint typecheck eval eval-fake docker clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package with dev extras
	pip install -e ".[dev]"

run:  ## Run the API on :8000 (offline fake providers by default)
	uvicorn groundwork.main:app --reload --port 8000

test:  ## Run the full test suite (no network, no API keys)
	pytest -q

test-cov:  ## Run tests with a coverage report
	pytest --cov=groundwork --cov-report=term-missing -q

lint:  ## Ruff lint + format check
	ruff check src tests && ruff format --check src tests

typecheck:  ## mypy
	mypy src

eval:  ## Run the benchmark against the CONFIGURED provider (costs money)
	python -m groundwork.evaluation.run_eval --out docs/eval-results.md

eval-fake:  ## Smoke-test the eval harness with no API cost
	python -m groundwork.evaluation.run_eval --provider fake --out /tmp/eval-smoke.md

docker:  ## Build and run with Postgres
	docker compose up --build

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__ groundwork.db
