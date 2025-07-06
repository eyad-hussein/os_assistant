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
4. Create a `.env` file with necessary variables. You can either copy the example file or create it manually.

   **On Linux/macOS:**

   ```bash
   cp .env.example .env
   ```
   **On Windows (CMD):**

   ```cmd
   copy .env.example .env
   ```

5. Edit `.env` and replace the value of `MODEL_BASE_URL` with your personal ngrok link, e.g.:

   ```
   MODEL_BASE_URL=https://your-ngrok-link.ngrok.io
   ```

6. Run the environment setup and model pulling script:

   **On Linux/macOS:**

   ```bash
   ./setup_and_pull.sh
   ```
   **On Windows (CMD):**

   ```cmd
   setup_and_pull.cmd
   ```

These scripts:

* Load variables from your `.env` file.
* Set the `OLLAMA_HOST` environment variable using `MODEL_BASE_URL`.
* Pull required models using `ollama pull`.
* 
## Usage
We primarily use [make](https://www.gnu.org/software/make/) as a command runner (bad practices ik). Have a look at the makefile for all available commands. 

## Build
Currently using pip for development build:
```bash
 $ uv pip install -e .
```
## CLI Usage
The `osassis` command line tool provides the following features:

### Interactive Chat
Open an interactive chat session with the OS assistant:
```bash
 $ osassis chat
```

### System Tracing
Start tracing file system events in a directory:
```bash
 $ osassis trace start file_system --dir path/to/watch
```

View trace logs with time filtering:
```bash
 $ osassis trace show file_system --start "yesterday" --end "now"
```

Clear trace logs for a domain:
```bash
 $ osassis trace clear file_system
```

## Testing with tox
tox creates virtual environments and runs all of pytest, ruff, and mypy.
```bash
 $ make tox
```
