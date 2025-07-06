import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

# Ensure the NGROK URL is correct or use your local Ollama endpoint

load_dotenv(override=True)
MODEL_BASE_URL = os.environ["MODEL_BASE_URL"]
MODEL_NAME = os.environ["MODEL_NAME"]
EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]

model = ChatOllama(model=MODEL_NAME, temperature=0, base_url=MODEL_BASE_URL)

# Create a backup model for fixing outputs
fixing_model = ChatOllama(model=MODEL_NAME, temperature=0, base_url=MODEL_BASE_URL)

# Define available domains
DOMAINS = ["file_system", "users", "packages", "network"]

# Define logs directory
LOGS_DIR = "domain_logs"

# Assistant mode configuration
# 0: Both code tool and RAG (default)
# 1: Code tool only (no RAG)
# 2: RAG only (no code tool)
# 3: Basic model only (no code tool, no RAG)
ASSISTANT_MODE = 1
