import os

# Model Configuration
LLM_MODEL = "llama3"
LLM_TEMPERATURE = 0
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://cc1d-34-145-73-84.ngrok-free.app")

# File paths
TEMP_EXECUTION_FILE = "temp_exec.py"
