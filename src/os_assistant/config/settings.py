import os

from dotenv import load_dotenv
from langchain_ollama import OllamaLLM

load_dotenv()
# # Ensure the NGROK URL is correct or use your local Ollama endpoint
# MODEL_BASE_URL = os.environ.get(
#     "OLLAMA_BASE_URL", "https://e115-35-247-70-88.ngrok-free.app"
# )
# MODEL_NAME = "llama3"
# MODEL_BASE_URL = "http://localhost:11434"
MODEL_NAME = os.getenv("MODEL_NAME")
MODEL_BASE_URL = os.getenv("MODEL_BASE_URL")
assert MODEL_NAME is not None and MODEL_BASE_URL is not None

model = OllamaLLM(model=MODEL_NAME, base_url=MODEL_BASE_URL)

# Create a backup model for fixing outputs
fixing_model = OllamaLLM(model=MODEL_NAME, base_url=MODEL_BASE_URL)

# Define available domains
DOMAINS = ["filesystem", "users", "packages", "network"]

# Define logs directory
LOGS_DIR = "domain_logs"
