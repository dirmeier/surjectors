.PHONY: tests, lints, docs, format

tests:
	uv run pytest

lints:
	uv run ruff check ramsey examples

format:
	uv run ruff check --fix surjectors examples
	uv run ruff format surjectors examples

docs:
	cd docs && make html
