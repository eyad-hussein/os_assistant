import os
from functools import lru_cache

import yaml

from dagent.utils import LOGGER, get_os_name
from dagent.utils.settings import ASSISTANT_MODE

# Default fallback prompt for error cases
DEFAULT_PROMPT = {
    "prompt": "Please provide information about: {prompt}",
    "system_message": "You are a helpful assistant.",
}

# Cache for mode configuration
_MODE_CONFIG_CACHE = None


def _load_mode_config():
    """Load mode configuration from YAML file"""
    global _MODE_CONFIG_CACHE

    if _MODE_CONFIG_CACHE is not None:
        return _MODE_CONFIG_CACHE

    config_path = "src/dagent/prompts/mode_config.yaml"
    try:
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
            # Convert mode_0, mode_1, etc. to integer keys
            _MODE_CONFIG_CACHE = {
                int(key.split("_")[1]): value for key, value in config.items()
            }
            return _MODE_CONFIG_CACHE
    except Exception as e:
        LOGGER.error(f"Error loading mode config: {str(e)}")
        # Fallback to basic mode config
        return {0: {}, 1: {}, 2: {}, 3: {}}


def load_prompt(node_name):
    """
    Returns a prompt dict where device info and mode-specific content are replaced,
    but other placeholders remain intact.
    """
    raw_data = _get_raw_prompt(node_name)

    # Copy to prevent cache mutation
    formatted_data = raw_data.copy()
    os_name = get_os_name()

    # Get mode-specific replacements from config file
    mode_config = _load_mode_config()
    mode_content = mode_config.get(ASSISTANT_MODE, mode_config.get(0, {}))

    # Apply replacements to both prompt and system_message
    for key in ["prompt", "system_message"]:
        if key in formatted_data:
            text = formatted_data[key]

            # Replace OS_NAME
            text = text.replace("{OS_NAME}", os_name)

            # Replace all mode-specific placeholders
            for placeholder, replacement in mode_content.items():
                text = text.replace("{" + placeholder + "}", replacement)

            formatted_data[key] = text

    return formatted_data


@lru_cache(maxsize=32)
def _get_raw_prompt(node_name):
    """Load a prompt from a YAML file from the common folder with caching

    This function is now cached to improve performance by avoiding repeated disk I/O
    for the same prompts. All prompts are now unified in the common folder.
    """
    # All prompts are now in the common folder
    prompt_path = f"src/dagent/prompts/{node_name}.yaml"
    try:
        if os.path.exists(prompt_path):
            with open(prompt_path, encoding="utf-8") as f:
                return yaml.safe_load(f)
        else:
            LOGGER.warning(f"Could not find prompt file {node_name} in common folder")
            return DEFAULT_PROMPT
    except Exception as e:
        LOGGER.error(f"Error loading prompt {node_name}: {str(e)}")
        return DEFAULT_PROMPT


def clear_prompt_cache():
    """Clear the prompt cache when needed (e.g., during development/testing)"""
    global _MODE_CONFIG_CACHE
    _get_raw_prompt.cache_clear()
    _MODE_CONFIG_CACHE = None
