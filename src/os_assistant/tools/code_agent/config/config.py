import os
import tempfile

from dotenv import load_dotenv

load_dotenv(override=True)
# Model Configuration
LLM_MODEL = os.environ["MODEL_NAME"]
LLM_MODEL_CODING = os.environ["CODING_AGENT_MODEL_NAME"]
LLM_TEMPERATURE = 0
OLLAMA_BASE_URL = os.environ["MODEL_BASE_URL"]
# File paths
TEMP_EXECUTION_FILE = "temp_exec.py"

# Output file configurations
OUTPUT_DIR = os.path.join(tempfile.gettempdir(), "os_assistant")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "execution_output.txt")
