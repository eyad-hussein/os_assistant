import os
from tracer.config import LogDomain

OLLAMA_BASE_URL = "https://34ae-35-243-144-241.ngrok-free.app"
EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_LLM_MODEL = "llama3" 

# Chunking settings
DEFAULT_CHUNK_SIZE = 100
DEFAULT_CHUNK_OVERLAP = 0.2 

# Database settings
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DB_DIR, exist_ok=True)

def get_db_path(domain: LogDomain = None):
    """Get domain-specific database path."""
    if domain:
        return os.path.join(DB_DIR, f"{domain.name.lower()}.sqlite")
    return os.path.join(DB_DIR, "logs_db.sqlite")  

# Retrieval settings
DEFAULT_TOP_K = 5

# Time formatting
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"  # ISO format 
