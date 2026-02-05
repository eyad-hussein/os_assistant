import os
import shutil
from pathlib import Path

from experiments.gepa.run_gepa import generate_variants, list_variants
from os_assistant.prompts.prompt_loader import load_prompt

VARIANTS_DIR = Path("experiments/gepa/variants")


def test_generate_variants_creates_files(tmp_path):
    # Ensure we have a clean variants directory
    if VARIANTS_DIR.exists():
        shutil.rmtree(VARIANTS_DIR)

    generate_variants(["command_generation_node"], prefixes=["strict_json", "safety_first"])

    assert VARIANTS_DIR.exists()
    files = list(VARi for VARi in VARIANTS_DIR.iterdir() if VARi.is_file())
    # Expect at least two variant files
    assert any("command_generation_node.strict_json.yaml" in f.name for f in files)
    assert any("command_generation_node.safety_first.yaml" in f.name for f in files)


def test_prompt_loader_variant_override(monkeypatch):
    # Ensure variant exists
    variant_name = "strict_json"
    monkeypatch.setenv("PROMPT_VARIANT", variant_name)

    prompt = load_prompt("command_generation_node")
    # The strict_json variant prompt contains JSON instruction
    assert "JSON" in prompt.get("prompt", "") or "Strict JSON" in prompt.get("system_message", "")

    monkeypatch.delenv("PROMPT_VARIANT", raising=False)


def test_prepare_final_result_records_variant(monkeypatch):
    monkeypatch.setenv("PROMPT_VARIANT", "safety_first")

    from os_assistant.core.nodes.result_preparation import prepare_final_result_node

    state = {
        "prompt": "Check disk",
        "original_prompt": "Check disk",
        "domains": ["file_system"],
        "domain_analysis": None,
        "query_type": None,
        "information_response": None,
        "command_response": None,
        "contexts": {},
    }

    new_state = prepare_final_result_node(state)
    assert new_state["final_result"].prompt_variant == "safety_first"

    monkeypatch.delenv("PROMPT_VARIANT", raising=False)


def test_code_agent_prompt_variant(monkeypatch):
    monkeypatch.setenv("PROMPT_VARIANT", "strict_json")
    from os_assistant.tools.code_agent.llm.prompt_loader import create_code_generation_prompt

    prompt = create_code_generation_prompt()
    text = prompt.template if hasattr(prompt, 'template') else str(prompt)
    assert "JSON" in text or "json" in text

    monkeypatch.delenv("PROMPT_VARIANT", raising=False)
