import os
from langchain_ollama import OllamaLLM

# Ensure the NGROK URL is correct or use your local Ollama endpoint
MODEL_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "https://34ae-35-243-144-241.ngrok-free.app") 
MODEL_NAME = "llama3"

model = OllamaLLM(model=MODEL_NAME, base_url=MODEL_BASE_URL)

# Create a backup model for fixing outputs
fixing_model = OllamaLLM(model=MODEL_NAME, base_url=MODEL_BASE_URL)

# Define available domains
DOMAINS = ["file_system", "users", "packages", "network"]

# Define logs directory
LOGS_DIR = "domain_logs"
