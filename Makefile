.PHONY: fmt lint lint-fix test tox clean run setup_and_pull

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

run:
	uv run osassis chat

setup_and_pull:
	uv run setup_and_pull_general.py

docker_build:
	docker build -t dagent_dev .
#   --build-arg FILE_ID=1TDkZPvfC_x1dh5C6EX8ayKTEdYfx1Wb3

docker_run:
	docker run -it --rm --name dagent_dev_container dagent_dev

streamlit_run:
	uv run streamlit run streamlit_app/app.py
