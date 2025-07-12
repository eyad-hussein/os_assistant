import os
from functools import lru_cache

import yaml

from os_assistant.utils.settings import ASSISTANT_MODE

# Map assistant mode to folder name
MODE_FOLDERS = {
    0: "full",  # Both RAG and tool
    1: "tool_only",  # Tool only
    2: "rag_only",  # RAG only
    3: "basic",  # Neither
}

# Default fallback prompt for error cases
DEFAULT_PROMPT = {
    "prompt": "Please provide information about: {prompt}",
    "system_message": "You are a helpful assistant.",
}


def get_prompt_folder():
    """Get the appropriate prompt folder based on assistant mode"""
    folder = MODE_FOLDERS.get(ASSISTANT_MODE, "full")
    return folder


@lru_cache(maxsize=32)
def load_prompt(prompt_name):
    """Load a prompt from a YAML file based on current assistant mode with caching

    This function is now cached to improve performance by avoiding repeated disk I/O
    for the same prompts.
    """
    # Get the correct folder based on mode
    folder = get_prompt_folder()

    # First try to load from the mode-specific folder
    mode_specific_path = f"src/os_assistant/prompts/{folder}/{prompt_name}.yaml"

    # Fallback to the root prompts directory if mode-specific file doesn't exist
    root_path = f"src/os_assistant/prompts/{prompt_name}.yaml"

    try:
        # Try mode-specific path first
        if os.path.exists(mode_specific_path):
            with open(mode_specific_path, encoding="utf-8") as f:
                return yaml.safe_load(f)
        # Fall back to root path
        elif os.path.exists(root_path):
            with open(root_path, encoding="utf-8") as f:
                return yaml.safe_load(f)
        else:
            print(
                f"Warning: Could not find prompt file {prompt_name} in either {folder} or root directory"
            )
            # Return the default prompt to prevent system failure
            return DEFAULT_PROMPT
    except Exception as e:
        print(f"Error loading prompt {prompt_name}: {str(e)}")
        # Return the default prompt to prevent system failure
        return DEFAULT_PROMPT


def clear_prompt_cache():
    """Clear the prompt cache when needed (e.g., during development/testing)"""
    load_prompt.cache_clear()
