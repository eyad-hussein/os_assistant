#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv


def run(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("Error: command not found:", cmd[0])
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}: {' '.join(cmd)}")
        sys.exit(e.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Cross-platform env loader and Ollama model puller"
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to .env file (default: .env)",
    )
    args = parser.parse_args()

    env_path = Path(args.env_file)
    if not env_path.exists():
        print(f".env file not found at: {env_path}")
        sys.exit(1)

    load_dotenv(dotenv_path=env_path, override=True)

    # Ensure ollama is available
    if shutil.which("ollama") is None:
        print("Error: 'ollama' CLI is not in PATH. Install it and try again.")
        sys.exit(1)

    # Set OLLAMA_HOST from MODEL_BASE_URL
    model_base_url = os.environ.get("MODEL_BASE_URL", "").strip()
    if not model_base_url:
        print(
            "Warning: MODEL_BASE_URL is not set in .env; OLLAMA_HOST will not be set."
        )
    else:
        os.environ["OLLAMA_HOST"] = model_base_url
        print(f"OLLAMA_HOST set to {os.environ['OLLAMA_HOST']}")

    # Required model names
    model_name = os.environ.get("MODEL_NAME", "").strip()
    coding_agent_model = os.environ.get("CODING_AGENT_MODEL_NAME", "").strip()
    embedding_model = os.environ.get("EMBEDDING_MODEL", "").strip()

    missing = [
        k
        for k, v in {
            "MODEL_NAME": model_name,
            "CODING_AGENT_MODEL_NAME": coding_agent_model,
            "EMBEDDING_MODEL": embedding_model,
        }.items()
        if not v
    ]

    if missing:
        print("Error: missing required env var(s): " + ", ".join(missing))
        sys.exit(1)

    # Pull models
    run(["ollama", "pull", model_name])
    run(["ollama", "pull", coding_agent_model])
    run(["ollama", "pull", embedding_model])

    print("All models pulled successfully.")


if __name__ == "__main__":
    main()
