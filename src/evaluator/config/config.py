import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Model Configuration
LLM_MODEL = os.environ.get("MODEL_JUDGE_NAME", "ollama/llama3")
LLM_BASE_URL = os.environ.get("MODEL_BASE_URL", "http://localhost:11434")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.0"))

# Evaluation Metrics Configuration
EVALUATION_METRICS = [
    {
        "name": "correctness",
        "weight": 0.4,
        "description": "Measures factual accuracy and appropriateness of commands/information for Windows 10",
        "prompt_file": "correctness_evaluator.yaml",
    },
    {
        "name": "completeness",
        "weight": 0.4,
        "description": "Measures how thoroughly all aspects of the query are addressed",
        "prompt_file": "completeness_evaluator.yaml",
    },
    {
        "name": "clarity",
        "weight": 0.2,
        "description": "Measures how clear, well-structured and understandable the response is",
        "prompt_file": "clarity_evaluator.yaml",
    },
]

# Scoring Thresholds
THRESHOLDS = {
    "excellent": 4.5,
    "good": 3.5,
    "acceptable": 2.5,
    "poor": 1.5,
}

# Path Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")
DATASETS_DIR = "datasets"
RESULTS_DIR = "evaluation_results"

# Ensure directories exist
for directory in [PROMPTS_DIR, DATASETS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Evaluation Settings
BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", "5"))
MAX_RETRIES = int(os.environ.get("EVAL_MAX_RETRIES", "2"))
CACHE_RESULTS = os.environ.get("EVAL_CACHE_RESULTS", "True").lower() == "true"

# Reporting Options
DETAILED_REPORTS = os.environ.get("EVAL_DETAILED_REPORTS", "True").lower() == "true"


# Metric Weights (for easy access)
def get_metric_weights() -> dict[str, float]:
    """Get a dictionary of metric weights for easy access."""
    return {metric["name"]: metric["weight"] for metric in EVALUATION_METRICS}


# Get available metrics
def get_available_metrics() -> list[str]:
    """Get a list of available metric names."""
    return [metric["name"] for metric in EVALUATION_METRICS]
