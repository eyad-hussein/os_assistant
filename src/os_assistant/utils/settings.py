# import os

# from dotenv import load_dotenv
# from langchain_ollama import ChatOllama

# # Ensure the NGROK URL is correct or use your local Ollama endpoint

# load_dotenv(override=True)
# MODEL_BASE_URL = os.environ["MODEL_BASE_URL"]
# MODEL_NAME = os.environ["MODEL_NAME"]
# EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]

# model = ChatOllama(model=MODEL_NAME, temperature=0, base_url=MODEL_BASE_URL)

# # Create a backup model for fixing outputs
# fixing_model = ChatOllama(model=MODEL_NAME, temperature=0, base_url=MODEL_BASE_URL)

# # Define available domains
# DOMAINS = ["file_system", "users", "packages", "network"]

# # Define logs directory
# LOGS_DIR = "domain_logs"

# # Assistant mode configuration
# # 0: Both code tool and RAG (default)
# # 1: Code tool only (no RAG)
# # 2: RAG only (no code tool)
# # 3: Basic model only (no code tool, no RAG)
# ASSISTANT_MODE = 1


import os
from pathlib import Path
import yaml
from dotenv import load_dotenv

load_dotenv(override=True)

# -------------------------------------------------------------------
# Environment variables

MODEL_BASE_URL = os.getenv("MODEL_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# -------------------------------------------------------------------
# Load YAML configuration

CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = CURRENT_DIR.parent / "configs" / "config.yaml"
CONFIG_PATH = Path(os.getenv("OS_ASSISTANT_CONFIG", DEFAULT_CONFIG_PATH))

if CONFIG_PATH.exists():
    with CONFIG_PATH.open("r") as f:
        CONFIG = yaml.safe_load(f)
else:
    CONFIG = {}

DOMAINS = CONFIG.get("DOMAINS", [])
LOGS_DIR = CONFIG.get("LOGS_DIR", "domain_logs")
ASSISTANT_MODE = int(CONFIG.get("ASSISTANT_MODE", 1))
TEMPERATURE = int(CONFIG.get("TEMPERATURE", 0))