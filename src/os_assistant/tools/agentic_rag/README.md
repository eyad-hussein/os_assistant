# Agentic RAG for System Logs
RAG for Graduation Project.
## Project Structure

```
agentic_rag/
├── src/
│   └── agentic_rag/
│       ├── application/           # Application-level functionality
│       │   ├── __init__.py
│       │   ├── init.py            # Database initialization
│       │   └── search.py          # Log search implementation
│       │
│       ├── config/                # Configuration settings
│       │   ├── __init__.py
│       │   └── config.py          # Constants and configuration
│       │
│       ├── core/                  # Core RAG functionality
│       │   ├── __init__.py
│       │   ├── agent_rag.py       # LLM-based agent for summarization
│       │   ├── chunking.py        # Text chunking utilities
│       │   ├── embedding.py       # Vector embedding generation
│       │   └── retrieval.py       # Semantic retrieval engine
│       │
│       ├── database/              # Data persistence
│       │   ├── __init__.py
│       │   └── database.py        # Database operations
│       │
│       ├── processor/             # Log processing
│       │   ├── __init__.py
│       │   └── batch_processor.py # Batch processing of logs
│       │
│       └── __init__.py            # Package initialization
│
├── data/                          # Storage for databases
│   └── *.sqlite                   # SQLite database files
│
└── main.py                        # CLI entry point
```

## Components Explained

### 1. Application Layer (`application/`)

- **`init.py`**: Handles database initialization by processing log streams from the tracer system. It coordinates chunking, embedding generation, and storage.
- **`search.py`**: Implements the search functionality with support for multi-domain queries, automatic database initialization, and optional AI summarization.

### 2. Configuration (`config/`)

- **`config.py`**: Central configuration file containing settings for Ollama API, embedding models, database paths, chunking parameters, and other constants.

### 3. Core Components (`core/`)

- **`agent_rag.py`**: Implements the AI agent using LangGraph and Ollama LLM for generating summaries from log entries.
- **`chunking.py`**: Manages splitting of log text into optimally sized, overlapping chunks for better semantic retrieval.
- **`embedding.py`**: Handles vector embedding generation using Ollama's embedding models, with utilities for similarity calculations.
- **`retrieval.py`**: Core retrieval engine that performs semantic similarity search and aggregates results from multiple domains.

### 4. Database Layer (`database/`)

- **`database.py`**: Manages SQLite database operations, including initialization, insertion, retrieval, and vector similarity searching.

### 5. Processing Layer (`processor/`)

- **`batch_processor.py`**: Handles batch processing of log entries from the tracer system, with support for incremental updates and batch size control.

### 6. CLI Interface (`main.py`)

- Command-line interface with `init` and `search` commands for initializing databases and searching logs.

## Installation and Setup

### Prerequisites

- Python 3.8+
- Ollama server running (with `nomic-embed-text` and `llama3` models)
- Access to system tracer logs

### Setup Steps

1. Clone the repository:
```bash
git clone [repository-url]
cd os_assistant/src/os_assistant/tools/Agentic_RAG
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure Ollama endpoint:
   - Edit `src/agentic_rag/config/config.py` and update `OLLAMA_BASE_URL` to point to your Ollama server

## Usage

### Initializing the Log Database

Initialize the database for filesystem logs:
```bash
python main.py init --domain file_system
```

Initialize for multiple domains:
```bash
python main.py init --domain file_system,networking --chunk-size 128 --overlap 0.2
```

### Searching Logs

Basic search:
```bash
python main.py search "file read operation"
```

Advanced search with summarization:
```bash
python main.py search "suspicious network connections" --domain networking,file_system --top-k 10 --summarize
```



## Building from Source

To build the project from source:
1. Build the package:
```bash
pip install -e .
```

