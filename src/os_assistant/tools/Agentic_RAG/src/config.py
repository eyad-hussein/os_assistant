import os

OLLAMA_BASE_URL = "https://794d-34-151-110-242.ngrok-free.app"
EMBEDDING_MODEL = "nomic-embed-text"

# Chunking settings
DEFAULT_CHUNK_SIZE = 4
DEFAULT_CHUNK_OVERLAP = 0.2 

# Database settings
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "logs_db.sqlite")

# Retrieval settings
DEFAULT_TOP_K = 5
