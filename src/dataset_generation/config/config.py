from dagent.utils.settings import EMBEDDING_MODEL, MODEL_BASE_URL, MODEL_NAME

# LLM Configuration
DATASET_LLM_MODEL = MODEL_NAME
DATASET_LLM_BASE_URL = MODEL_BASE_URL
DATASET_EMBEDDING_MODEL = EMBEDDING_MODEL

# Dataset Generation Parameters
NUM_SAMPLES_PER_DOMAIN = 20
MIN_QUESTIONS_PER_LOG = 2
MAX_QUESTIONS_PER_LOG = 5

# Log Sampling Parameters
MIN_SEQUENTIAL_LOGS = 3  # Minimum number of sequential logs to sample together
MAX_SEQUENTIAL_LOGS = 5  # Maximum number of sequential logs to sample together
TIME_WINDOW_SECONDS = (
    300  # Consider logs sequential if within this time window (5 minutes)
)


# Similarity Settings
SIMILARITY_THRESHOLD = 0.99  # Threshold for considering questions as duplicates
VECTOR_CACHE_SIZE = 1000  # Number of embeddings to cache

# Output Settings
DATASET_OUTPUT_DIR = "datasets"
DEFAULT_DATASET_FILENAME = "os_command_dataset.json"
