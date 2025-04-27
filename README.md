[![license](https://img.shields.io/badge/license-MIT-blue)](https://opensource.org/license/mit/)
[![Tests](https://github.com/omar-abdelgawad/python-project-template/actions/workflows/tests.yml/badge.svg)](https://github.com/omar-abdelgawad/python-project-template/actions)
[![PythonVersion](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)
<!-- [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) -->

# os_assistant
This is a modern template for a python project with the pyproject.toml with some fields to change based on project. It also has tox configured, docs dir for github pages, .github dir with tox-gh-actions configured and more.  

## Prerequisites
Make sure [uv](https://docs.astral.sh/uv/getting-started/installation/#installation-methods) is installed. Here is the install script on macOS and Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Installation
1. clone the repo and cd into it.
```bash
 $ git clone <repo_url>
 $ cd os_assistant
```
2. run `uv sync` to install all dependencies in a virtual environment. Note that this step is not necessary as uv automatically runs it before running any script with `uv run`.
```bash
 $ uv sync
```
3. Make sure that pre-commit hooks are installed.
```bash
 $ pre-commit install
``` 
4. Create a `.env` file with necessary variables. You can use the default settings through the following command.
```bash
 $ mv .env.example .env
```

## Usage
We primarily use [make](https://www.gnu.org/software/make/) as a command runner (bad practices ik). Have a look at the makefile for all available commands. 

## Testing with tox
tox creates virtual environments and runs all of pytest, ruff, and mypy.
```bash
 $ make tox
```
