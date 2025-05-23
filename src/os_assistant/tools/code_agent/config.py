import os

from dotenv import load_dotenv

load_dotenv(override=True)
# Model Configuration
LLM_MODEL = os.environ["MODEL_NAME"]
LLM_TEMPERATURE = 0
OLLAMA_BASE_URL = os.environ["MODEL_BASE_URL"]
# File paths
TEMP_EXECUTION_FILE = "temp_exec.py"
