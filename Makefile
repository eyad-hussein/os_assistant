fmt:
	uv run ruff format

lint:
	uv run ruff check

lint-fix:
	uv run ruff check --fix

test:
	uv run pytest

tox:
	uv run tox

demo:
	uv run examples/os_assistant.py
