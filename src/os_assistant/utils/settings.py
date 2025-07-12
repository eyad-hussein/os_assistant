import os
from pathlib import Path
import yaml
from dotenv import load_dotenv

load_dotenv(override=True)

# -------------------------------------------------------------------
# Environment variables

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

CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = CURRENT_DIR.parent / "configs" / "config.yaml"
CONFIG_PATH = Path(os.getenv("OS_ASSISTANT_CONFIG", DEFAULT_CONFIG_PATH))

if CONFIG_PATH.exists():
    with CONFIG_PATH.open("r") as f:
        CONFIG = yaml.safe_load(f)
else:
    CONFIG = {}

DOMAINS = CONFIG.get("DOMAINS", [])
LOGS_DIR = CONFIG.get("LOGS_DIR", "domain_logs")
ASSISTANT_MODE = int(CONFIG.get("ASSISTANT_MODE", 1))
TEMPERATURE = int(CONFIG.get("TEMPERATURE", 0))
GRAPH_VISUALIZE = bool(CONFIG.get("GRAPH_VISUALIZE", False))
