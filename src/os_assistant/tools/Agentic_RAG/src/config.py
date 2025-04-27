import os

OLLAMA_BASE_URL = "https://d73e-35-204-201-37.ngrok-free.app"
EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_LLM_MODEL = "llama3"  # Llama3 model for summarization

# Chunking settings
DEFAULT_CHUNK_SIZE = 4
DEFAULT_CHUNK_OVERLAP = 0.2 

# Database settings
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "logs_db.sqlite")

# Retrieval settings
DEFAULT_TOP_K = 5
