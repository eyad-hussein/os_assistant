import os

from tracer.config import LogDomain

from os_assistant.utils.settings import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL,
    MODEL_BASE_URL,
    MODEL_NAME,
    TIMESTAMP_FORMAT,
)

OLLAMA_BASE_URL = MODEL_BASE_URL
EMBEDDING_MODEL = EMBEDDING_MODEL
OLLAMA_LLM_MODEL = MODEL_NAME

# Chunking settings
DEFAULT_CHUNK_SIZE = DEFAULT_CHUNK_SIZE
DEFAULT_CHUNK_OVERLAP = DEFAULT_CHUNK_OVERLAP
DEFAULT_TOP_K = DEFAULT_TOP_K
# Time formatting
TIMESTAMP_FORMAT = TIMESTAMP_FORMAT

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
