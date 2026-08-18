.PHONY: install lint typecheck test quality notebook

install:
	poetry install

lint:
	poetry run ruff check src tests

typecheck:
	poetry run mypy src

test:
	poetry run pytest

quality: lint typecheck test

notebook:
	poetry run jupyter lab
