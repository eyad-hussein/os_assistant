#!/usr/bin/env bash
set -e

# Load .env into environment
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs -d '\n')
else
  echo ".env file not found. Exiting."
  exit 1
fi

# Set OLLAMA_HOST
export OLLAMA_HOST="$MODEL_BASE_URL"
echo "OLLAMA_HOST set to $OLLAMA_HOST"

# Pull models
ollama pull "$MODEL_NAME"
ollama pull "$CODING_AGENT_MODEL_NAME"
ollama pull "$EMBEDDING_MODEL"
