import os


# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Model Configuration
LLM_MODEL = "claude-3-sonnet-20240229"
LLM_TEMPERATURE = 0

# File paths
TEMP_EXECUTION_FILE = "temp_exec.py"
