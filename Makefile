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

# TODO: the following clean command would also clean the virtual environment so don't use it
clean:
	git clean -fxfd -e '*venv*' --dry-run

demo:
	uv run examples/os_assistant.py
