import os

from os_assistant.utils.settings import (
    CODING_AGENT_MODEL_NAME,
    MODEL_BASE_URL,
    MODEL_NAME,
    TEMPERATURE,
)

# Model Configuration
LLM_MODEL = MODEL_NAME
LLM_MODEL_CODING = CODING_AGENT_MODEL_NAME
LLM_TEMPERATURE = TEMPERATURE
OLLAMA_BASE_URL = MODEL_BASE_URL

# Directory configuration
CWD = os.getcwd()
OUTPUT_DIR = os.path.join(CWD, "outputs")

# Execution configuration
TEMP_EXECUTION_FILE = "temp_execution.py"
MAX_CONSECUTIVE_ERRORS = 5
