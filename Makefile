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

clean:
	git clean -fxfd -e '*venv*' -e ".env" --dry-run

demo:
	uv run examples/Experimental_Code_lang/main.py

run:
	uv run osassis chat

setup_and_pull:
	uv run setup_and_pull_general.py
