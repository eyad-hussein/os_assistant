import os
from functools import lru_cache

import yaml

from os_assistant.utils import LOGGER, get_os_name
from os_assistant.utils.settings import ASSISTANT_MODE

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

    config_path = "src/os_assistant/prompts/mode_config.yaml"
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
    but other placeholders remain intact. Supports optional GEPA prompt variants.

    Behavior:
    - Load base prompt from prompts folder
    - If environment variable PROMPT_VARIANT is set, attempt to load
      experiments/gepa/variants/{node_name}.{PROMPT_VARIANT}.yaml and use it to
      override `prompt` and/or `system_message` fields.
    """
    # Load the base prompt
    raw_data = _get_raw_prompt(node_name)

    # Copy to prevent cache mutation
    formatted_data = raw_data.copy()
    os_name = get_os_name()

    # Check for GEPA variant override
    # Node-specific variant mapping support
    # Priority: explicit PROMPT_VARIANT env -> per-node mapping file -> GEPA_PROMPT_VARIANT global
    variant_name = os.getenv("PROMPT_VARIANT") or os.getenv("GEPA_PROMPT_VARIANT")

    # Check for a per-node mapping file (optional)
    variant_map_path = os.getenv("PROMPT_VARIANT_MAP_PATH", "experiments/gepa/variant_map.json")
    try:
        if os.path.exists(variant_map_path):
            with open(variant_map_path, encoding="utf-8") as vmf:
                vm = yaml.safe_load(vmf)
                if isinstance(vm, dict) and node_name in vm:
                    # Use node-specific mapping
                    variant_name = vm.get(node_name)
    except Exception as e:
        LOGGER.debug(f"Could not load PROMPT_VARIANT_MAP_PATH at {variant_map_path}: {e}")

    if variant_name:
        # First try node-specific variant file
        variant_path = f"experiments/gepa/variants/{node_name}.{variant_name}.yaml"
        # Then try a generic node file with variant folder (fallback)
        variant_path_fallback = f"experiments/gepa/variants/{node_name}.yaml"

        try:
            if os.path.exists(variant_path):
                with open(variant_path, encoding="utf-8") as vf:
                    variant_data = yaml.safe_load(vf)
                    # Merge override keys (only replace provided fields)
                    if isinstance(variant_data, dict):
                        for k, v in variant_data.items():
                            formatted_data[k] = v
            elif os.path.exists(variant_path_fallback):
                with open(variant_path_fallback, encoding="utf-8") as vf:
                    variant_data = yaml.safe_load(vf)
                    if isinstance(variant_data, dict):
                        for k, v in variant_data.items():
                            formatted_data[k] = v
            else:
                # No variant file found - continue with base prompt
                pass
        except Exception as e:
            LOGGER.error(f"Error loading prompt variant {variant_name} for {node_name}: {e}")

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


def list_prompt_variants(node_name: str | None = None) -> list[str]:
    """List available GEPA prompt variants for a node or all nodes.

    Looks for files in `experiments/gepa/variants/` with pattern
    `<node_name>.<variant>.yaml` or `<node_name>.yaml`.
    """
    variants_dir = os.path.join("experiments", "gepa", "variants")
    if not os.path.exists(variants_dir):
        return []

    results = []
    for fname in os.listdir(variants_dir):
        if not fname.endswith(".yaml") and not fname.endswith(".yml"):
            continue
        parts = fname.rsplit(".", 2)
        # Possible patterns: node.variant.yaml or node.yaml
        if len(parts) == 3:
            node, variant, _ = parts
            if node_name and node != node_name:
                continue
            results.append(f"{node}.{variant}")
        else:
            node = parts[0]
            if node_name and node != node_name:
                continue
            results.append(node)

    return results


@lru_cache(maxsize=32)
def _get_raw_prompt(node_name):
    """Load a prompt from a YAML file from the common folder with caching

    This function is now cached to improve performance by avoiding repeated disk I/O
    for the same prompts. All prompts are now unified in the common folder.
    """
    # All prompts are now in the common folder
    prompt_path = f"src/os_assistant/prompts/{node_name}.yaml"
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
