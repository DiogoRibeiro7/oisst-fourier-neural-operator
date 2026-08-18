.PHONY: help install hooks lint format format-check typecheck test quality notebook build clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

install:  ## Install the project and dev dependencies
	poetry install

hooks:  ## Install the pre-commit git hooks
	poetry run pre-commit install

lint:  ## Run ruff
	poetry run ruff check src tests

format:  ## Apply ruff formatting
	poetry run ruff format src tests

format-check:  ## Verify formatting without writing
	poetry run ruff format --check src tests

typecheck:  ## Run mypy in strict mode
	poetry run mypy src

test:  ## Run the test suite
	poetry run pytest

quality: lint format-check typecheck test  ## Run every check CI runs

notebook:  ## Launch JupyterLab
	poetry run jupyter lab

build:  ## Build the wheel and sdist
	poetry build

clean:  ## Remove build and cache artifacts
	rm -rf dist build .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
