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
```bash
 $ git clone <repo_url>
 $ cd os_assistant
 $ uv sync
``` 

## Usage
We primarily use make as a command runner (bad practices ik). Have a look at the makefile for all available commands. 

## Testing with tox
Just running `tox` with no args should work.
```bash
 $ make tox
```
tox creates virtual environments and runs all of pytest, ruff, and mypy.
