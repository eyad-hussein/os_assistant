import os

import yaml
from langchain_core.prompts import PromptTemplate

_PROMPTS = None


def _load_prompts():
    """Load all prompts from YAML file"""
    global _PROMPTS
    if _PROMPTS is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_file = os.path.join(current_dir, "prompts.yaml")

        with open(prompt_file, encoding="utf-8") as f:
            _PROMPTS = yaml.safe_load(f)

    # Apply optional GEPA variant overrides for code agent prompts
    variant = os.getenv("PROMPT_VARIANT") or os.getenv("GEPA_PROMPT_VARIANT")
    variant_file = os.path.join(os.path.dirname(current_dir), "..", "..", "..", "experiments", "gepa", "variants", f"code_agent.{variant}.yaml") if variant else None
    try:
        if variant and os.path.exists(variant_file):
            with open(variant_file, encoding="utf-8") as vf:
                variant_data = yaml.safe_load(vf)
                # Merge overrides into prompts
                if isinstance(variant_data, dict):
                    for key, val in variant_data.items():
                        if key in _PROMPTS and isinstance(val, dict):
                            _PROMPTS[key].update(val)
                        else:
                            _PROMPTS[key] = val
    except Exception:
        # If any error occurs while loading variants, log and continue with base prompts
        pass

    return _PROMPTS


def get_prompt(prompt_name):
    """Get a prompt by name and convert to PromptTemplate"""
    prompts = _load_prompts()
    if prompt_name not in prompts:
        raise ValueError(f"Prompt '{prompt_name}' not found")

    prompt_data = prompts[prompt_name]
    return PromptTemplate.from_template(prompt_data["prompt"])


def create_code_generation_prompt():
    """Get the code generation prompt"""
    return get_prompt("code_generation")


def create_code_error_prompt():
    """Get the code error prompt"""
    return get_prompt("code_error")


def create_summary_prompt():
    """Get the summary prompt"""
    return get_prompt("summary")
