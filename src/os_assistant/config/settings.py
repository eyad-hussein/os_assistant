import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

# Ensure the NGROK URL is correct or use your local Ollama endpoint

load_dotenv()
MODEL_BASE_URL = os.environ["MODEL_BASE_URL"]
MODEL_NAME = os.environ["MODEL_NAME"]

model = ChatOllama(model=MODEL_NAME, base_url=MODEL_BASE_URL)

# Create a backup model for fixing outputs
fixing_model = ChatOllama(model=MODEL_NAME, base_url=MODEL_BASE_URL)

# Define available domains
DOMAINS = ["file_system", "users", "packages", "network"]

# Define logs directory
LOGS_DIR = "domain_logs"
