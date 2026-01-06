import os
from functools import lru_cache

import yaml

from os_assistant.utils import LOGGER, get_os_name
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

COMMON_NODES = [
    "conversation_context_node",
]


def get_prompt_folder(node_name):
    """Get the appropriate prompt folder"""
    if node_name in COMMON_NODES:
        return "common"
    folder = MODE_FOLDERS.get(ASSISTANT_MODE, "full")
    return folder


def load_prompt(node_name):
    """
    Returns a prompt dict where device info is already replaced,
    but other placeholders remain intact.
    """
    raw_data = _get_raw_prompt(node_name)

    # Copy to prevent cache mutation
    formatted_data = raw_data.copy()
    os_name = get_os_name()

    # Safely swap the text without touching other {brackets}
    if "prompt" in formatted_data:
        formatted_data["prompt"] = formatted_data["prompt"].replace(
            "{OS_NAME}", os_name
        )

    return formatted_data


@lru_cache(maxsize=32)
def _get_raw_prompt(node_name):
    """Load a prompt from a YAML file based on current assistant mode with caching

    This function is now cached to improve performance by avoiding repeated disk I/O
    for the same prompts.
    """
    # Get the correct folder based on mode
    folder = get_prompt_folder(node_name)

    # First try to load from the mode-specific folder
    mode_specific_path = f"src/os_assistant/prompts/{folder}/{node_name}.yaml"
    try:
        if os.path.exists(mode_specific_path):
            with open(mode_specific_path, encoding="utf-8") as f:
                return yaml.safe_load(f)
        else:
            LOGGER.warning(f"Could not find prompt file {node_name} in {folder}")
            return DEFAULT_PROMPT
    except Exception as e:
        LOGGER.error(f"Error loading prompt {node_name}: {str(e)}")
        return DEFAULT_PROMPT


def clear_prompt_cache():
    """Clear the prompt cache when needed (e.g., during development/testing)"""
    load_prompt.cache_clear()
