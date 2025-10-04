import os

from dotenv import load_dotenv

from ..configs import CONFIG

load_dotenv(override=True)

# -------------------------------------------------------------------
# Environment variables
MODEL_TYPE = os.getenv("MODEL_TYPE")
MODEL_BASE_URL = os.getenv("MODEL_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# Code Agent Configuration
LLM_MODEL_CODING = os.getenv("CODING_AGENT_MODEL_NAME")

# Code Agent Directory configuration
CWD = os.getcwd()
OUTPUT_DIR = os.path.join(CWD, "outputs")

# Code Agent Execution configuration
TEMP_EXECUTION_FILE = "temp_execution.py"
MAX_CONSECUTIVE_ERRORS = 5

# -------------------------------------------------------------------
# Load YAML configuration

DOMAINS = CONFIG.get("DOMAINS", [])
LOGS_DIR = CONFIG.get("LOGS_DIR", "domain_logs")
ASSISTANT_MODE = int(CONFIG.get("ASSISTANT_MODE", 1))
TEMPERATURE = int(CONFIG.get("TEMPERATURE", 0))
GRAPH_VISUALIZE = bool(CONFIG.get("GRAPH_VISUALIZE", False))

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
