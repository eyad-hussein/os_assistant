import os

from dotenv import load_dotenv

load_dotenv(override=True)
# Model Configuration
LLM_MODEL = os.environ["MODEL_NAME"]
LLM_MODEL_CODING = os.environ["CODING_AGENT_MODEL_NAME"]
LLM_TEMPERATURE = 0
OLLAMA_BASE_URL = os.environ["MODEL_BASE_URL"]

# Directory configuration
CWD = os.getcwd()
OUTPUT_DIR = os.path.join(CWD, "outputs")

# Execution configuration
TEMP_EXECUTION_FILE = "temp_execution.py"
MAX_CONSECUTIVE_ERRORS = 5
