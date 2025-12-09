# NOTE: Only build after you get the setup working locally. Files such as .env
# files must first be created locally so that it will be copied to the image.
# This is because the host link from ngrok changes each time for a new session
# so we have to copy the latest during build.
# The Folder we want to have locally is stored with
# ==========================================================
# Stage 1: Base setup
# ==========================================================
FROM python:3.12-slim AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin:${PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    gnupg \
    unrar-free \
    git \
    && rm -rf /var/lib/apt/lists/*

# ==========================================================
# Install Ollama
# ==========================================================
# Official method: https://ollama.com/download/linux
RUN wget https://ollama.com/download/ollama-linux-amd64 -O /usr/local/bin/ollama && \
    chmod +x /usr/local/bin/ollama

# ==========================================================
# Stage 2: Setup project environment
# ==========================================================
WORKDIR /app

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Copy project files
COPY . .

# ==========================================================
# Stage 3: Download and extract data
# ==========================================================
# ID of Google Drive file ID (compressed .rar file)
ARG FILE_ID=1b25qc2LLGjc_FPyqbcpbKkve_VhtxS5a
ARG FILE_NAME=data.rar

# Create a directory for the data
RUN mkdir -p /home/user && \
    echo "Downloading .rar from Google Drive..." && \
    curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /dev/null && \
    CONFIRM=$(awk '/_warning_/ {print $NF}' /tmp/cookie) && \
    curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CONFIRM}&id=${FILE_ID}" -o "/home/user/${FILE_NAME}" && \
    echo "Extracting..." && \
    unrar-free x -y "/home/user/${FILE_NAME}" /home/user && \
    rm "/home/user/${FILE_NAME}"

# ==========================================================
# Stage 4: Python environment setup
# ==========================================================
# Sync dependencies (creates virtual environment)
RUN uv sync
# ==========================================================
# Default behavior
# ==========================================================
CMD [ "bash" ]
