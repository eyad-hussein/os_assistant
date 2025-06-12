import os
import tempfile

from dotenv import load_dotenv

load_dotenv(override=True)
# Model Configuration
LLM_MODEL = os.environ["MODEL_NAME"]
LLM_MODEL_CODING = os.environ["CODING_AGENT_MODEL_NAME"]
LLM_TEMPERATURE = 0
OLLAMA_BASE_URL = os.environ["MODEL_BASE_URL"]

# Directory and file configurations
# Use current working directory for output files instead of temp directories
CWD = os.getcwd()
OUTPUT_DIR = os.path.join(CWD, "outputs")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "execution_output.txt")
RESULTS_FILE = os.path.join(OUTPUT_DIR, "results.txt")

# Execution configuration
TEMP_EXECUTION_FILE = "temp_execution.py"  # This will be created in current directory
MAX_CONSECUTIVE_ERRORS = 5
