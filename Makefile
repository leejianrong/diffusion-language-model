.DEFAULT_GOAL := help
.PHONY: help install fmt lint typecheck test check hooks

help: ## List targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

install: ## Install dependencies from the lockfile (CPU torch)
	uv sync --frozen

fmt: ## Format and autofix
	uv run ruff format .
	uv run ruff check --fix .

lint: ## Lint and format check (no changes)
	uv run ruff check .
	uv run ruff format --check .

typecheck: ## Strict type check
	uv run mypy

test: ## Fast no-infra tests (CPU, seconds)
	uv run pytest tests/unit -q

check: lint typecheck test ## The pre-push gate: lint + types + fast tests

hooks: ## Install the pre-push hook
	ln -sf ../../scripts/git-hooks/pre-push .git/hooks/pre-push
