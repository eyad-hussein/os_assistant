import os

from dotenv import load_dotenv
from tracer.config import LogDomain

load_dotenv(override=True)
OLLAMA_BASE_URL = os.environ["MODEL_BASE_URL"]
EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]
OLLAMA_LLM_MODEL = os.environ["MODEL_NAME"]

# Chunking settings
DEFAULT_CHUNK_SIZE = 100
DEFAULT_CHUNK_OVERLAP = 0.2

# Database settings
DB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)
os.makedirs(DB_DIR, exist_ok=True)


def get_db_path(domain: LogDomain | None = None):
    """Get domain-specific database path."""
    if domain:
        return os.path.join(DB_DIR, f"{domain.name.lower()}.sqlite")
    return os.path.join(DB_DIR, "logs_db.sqlite")


# Retrieval settings
DEFAULT_TOP_K = 5

# Time formatting
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"  # ISO format
