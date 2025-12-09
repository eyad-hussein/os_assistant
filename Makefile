.PHONY: fmt lint lint-fix test tox clean demo run setup_and_pull

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

docker_build:
	docker build -t os_assistant_dev .
#   --build-arg FILE_ID=1TDkZPvfC_x1dh5C6EX8ayKTEdYfx1Wb3

docker_run:
	docker run -it --rm --name os_assistant_dev_container os_assistant_dev