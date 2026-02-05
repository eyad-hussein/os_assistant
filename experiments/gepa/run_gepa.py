"""GEPA prompt generation and evaluation runner for OS Assistant

This script provides the following utilities:
- generate variants heuristically for nodes
- list available variants
- evaluate variants on the provided dataset (dry-run heuristics or real model if configured)

If `dspy` is available, this script can be extended to orchestrate formal GEPA experiments.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import yaml

from os_assistant.prompts.prompt_loader import list_prompt_variants, load_prompt
from os_assistant.utils import LOGGER

VARIANTS_DIR = Path("experiments/gepa/variants")
DATASET_PATH = Path("experiments/gepa/dataset.jsonl")

# Heuristic generator: creates a small set of variants per node
DEFAULT_PREFIXES = ["strict_json", "safety_first", "few_shot"]


def generate_variants(nodes: List[str], prefixes: List[str] | None = None):
    prefixes = prefixes or DEFAULT_PREFIXES
    VARIANTS_DIR.mkdir(parents=True, exist_ok=True)

    for node in nodes:
        # Load base prompt if exists
        try:
            base = load_prompt(node)
        except Exception:
            base = {"prompt": f"Please provide information about: {{prompt}}", "system_message": "You are a helpful assistant."}

        for p in prefixes:
            fname = VARIANTS_DIR / f"{node}.{p}.yaml"
            if fname.exists():
                LOGGER.info(f"Variant {fname} already exists, skipping")
                continue

            # Create simple variant transformations
            if p == "strict_json":
                variant = {
                    "prompt": """
                    You must respond with a single JSON object following the node schema. Provide only the JSON object and nothing else. Example: {\"command\": \"echo hi\"}
                    """,
                    "system_message": base.get("system_message", "You are a helpful assistant.") + "\nStrict JSON required."
                }
            elif p == "safety_first":
                variant = {
                    "prompt": """
                    Prioritize safety. If any action could be destructive, refuse and provide a safe alternative in the response's `security_notes` field.
                    """,
                    "system_message": base.get("system_message", "") + "\nSafety-first instruction enabled."
                }
            else:  # few_shot
                variant = {
                    "prompt": base.get("prompt", "Please provide information about: {prompt}") + "\n\nExample:\nInput: 'Find largest files in /var/log'\nOutput: {\"command\": \"du -ah /var/log | sort -rh | head -n 20\"}",
                    "system_message": base.get("system_message", "")
                }

            with open(fname, "w", encoding="utf-8") as f:
                yaml.safe_dump(variant, f)
            LOGGER.info(f"Generated variant: {fname}")


def list_variants(node: str | None = None):
    variants = list_prompt_variants(node)
    if not variants:
        LOGGER.info("No prompt variants found.")
        return

    for v in variants:
        LOGGER.info(v)


def _load_dataset(path: Path = DATASET_PATH):
    if not Path(path).exists():
        LOGGER.warning(f"Dataset not found at {path}, returning empty list")
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def evaluate_variant(node_variant: str, dry_run: bool = True, dataset_path: Path | None = None):
    """Evaluate a specific variant. node_variant is like 'command_generation_node.strict_json'"""
    parts = node_variant.split(".")
    if len(parts) < 2:
        raise ValueError("node_variant should be in the format <node>.<variant>")
    node = parts[0]
    variant = parts[1]

    os.environ["PROMPT_VARIANT"] = variant

    ds = _load_dataset(Path(dataset_path) if dataset_path else DATASET_PATH)
    relevant = [r for r in ds if r.get("node") == node]

    if not relevant:
        LOGGER.warning(f"No dataset entries found for node {node}")
        return {"evaluated": 0}

    results = []
    for rec in relevant:
        # Get formatted prompt
        p = load_prompt(node)
        prompt_template = p.get("prompt", "{prompt}")
        # Simple formatting
        formatted = prompt_template.replace("{prompt}", rec.get("prompt", ""))

        # Dry-run heuristics: check that the prompt demands JSON or safety
        score = 0.0
        if "only" in formatted.lower() and "json" in formatted.lower():
            score += 0.7
        if "safety" in formatted.lower() or "refuse" in formatted.lower():
            score += 0.2
        if "example" in formatted.lower():
            score += 0.1

        results.append({"id": rec.get("id"), "score": score, "formatted_prompt_preview": formatted[:200]})

    # Clear variant env for cleanliness
    del os.environ["PROMPT_VARIANT"]

    # Simple aggregation
    avg_score = sum(r["score"] for r in results) / len(results)
    return {"evaluated": len(results), "avg_score": avg_score, "details": results}


def main():
    parser = argparse.ArgumentParser("GEPA prompt runner")
    parser.add_argument("--generate-variants", action="store_true")
    parser.add_argument("--list-variants", action="store_true")
    parser.add_argument("--nodes", nargs="*", help="Nodes to operate on (defaults to main nodes)")
    parser.add_argument("--evaluate", help="Evaluate a specific node.variant, e.g., command_generation_node.strict_json")
    parser.add_argument("--dry-run", action="store_true", help="Do not invoke real model; use heuristics")
    parser.add_argument("--gepa-run", action="store_true", help="Run a full GEPA experiment using dspy if available")
    parser.add_argument("--gepa-config", help="Optional path to GEPA config YAML to override defaults")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark comparing baseline vs specified GEPA variants")
    parser.add_argument("--variants", nargs="*", help="List of GEPA variant names to benchmark (e.g., strict_json safety_first). If omitted, all discovered variants are used.")
    parser.add_argument("--use-evaluator", action="store_true", help="Use the OSAssistantEvaluator (LLM judge) for benchmark scoring")
    parser.add_argument("--dataset-file", help="Optional dataset JSON file path to pass to the evaluator (overrides internal dataset)")
    parser.add_argument("--repeats", type=int, default=1, help="Number of repeats per sample for benchmarks (default: 1)")
    parser.add_argument("--seed", type=int, default=None, help="Base seed for deterministic repeat runs (optional)")

    args = parser.parse_args()

    # Determine dataset path (allow test_env defaults)
    dataset_path = DATASET_PATH
    if args.dataset_file:
        dataset_path = Path(args.dataset_file)
    else:
        # If a test_env dataset exists, prefer it
        testenv_dataset = Path("test_env") / "dataset.jsonl"
        if testenv_dataset.exists():
            dataset_path = testenv_dataset
        else:
            if not dataset_path.exists():
                LOGGER.warning(f"Default GEPA dataset not found at {dataset_path}. Consider passing --dataset-file or placing a dataset at {testenv_dataset}")

    def _validate_environment(path: Path) -> bool:
        # Basic checks: dataset presence and default test_env/data folder (if applicable)
        ok = True
        if not Path(path).exists():
            LOGGER.warning(f"Dataset not found at {path}. Benchmarks will be skipped or use empty dataset.")
            ok = False
        # If dataset lives under test_env, confirm test_env/data exists
        try:
            if "test_env" in str(path.resolve().parts):
                test_data_dir = Path("test_env") / "data"
                if not test_data_dir.exists():
                    LOGGER.warning(f"test_env/data not found at {test_data_dir}. Please ensure your dataset and tracer logs are placed under test_env/data.")
                    ok = False
        except Exception:
            pass
        return ok

    # If requested, run a full GEPA experiment
    if args.gepa_run:
        try:
            # Validate environment and dataset
            _validate_environment(dataset_path)

            from . import gepa_adapter

            # Load default experiment config
            base_config = yaml.safe_load(open(Path(__file__).parent / "config.yaml", encoding="utf-8"))
            ge_config = {"metrics": base_config.get("metrics", {})}

            # Add dataset and variants
            ge_config["dataset_path"] = str(dataset_path)
            ge_config["variants_dir"] = str(VARIANTS_DIR)
            ge_config["results_dir"] = str(Path("experiments") / "gepa" / "results")

            # Merge any user config file
            if args.gepa_config:
                try:
                    user_conf = yaml.safe_load(open(args.gepa_config, encoding="utf-8"))
                    if isinstance(user_conf, dict):
                        ge_config.update(user_conf)
                except Exception as e:
                    LOGGER.error(f"Could not load GEPA config file {args.gepa_config}: {e}")

            LOGGER.info("Starting GEPA experiment via dspy...")
            res = gepa_adapter.run_gepa_experiment(ge_config)
            print(json.dumps(res, indent=2))
        except Exception as e:
            LOGGER.error(f"GEPA run failed: {e}")
        return

    nodes = args.nodes or [
        "command_generation_node",
        "information_generation_node",
        "query_classifier_node",
        "domain_analysis_node",
    ]

    if args.generate_variants:
        generate_variants(nodes)

    if args.list_variants:
        list_variants()

    if args.evaluate:
        res = evaluate_variant(args.evaluate, dry_run=args.dry_run, dataset_path=dataset_path)
        print(json.dumps(res, indent=2))

    if args.benchmark:
        # Determine variants to run; allow user override via --variants
        variants = None
        if args.variants and len(args.variants) > 0:
            variants = args.variants
        else:
            # Support passing a single node.variant via --nodes for convenience
            if args.nodes and len(args.nodes) == 1 and "." in args.nodes[0]:
                parts = args.nodes[0].split(".")
                if len(parts) >= 2:
                    variants = [parts[1]]
            elif args.nodes:
                variants = args.nodes

        try:
            from .benchmark import BenchmarkRunner

            # Validate environment/dataset and warn if issues
            _validate_environment(dataset_path)

            runner = BenchmarkRunner(dataset_path)
            benchmark_res = runner.run_head_to_head(variants, use_evaluator=args.use_evaluator, evaluator_dataset=args.dataset_file, repeats=args.repeats, seed=args.seed)
            print(json.dumps(benchmark_res, indent=2))
        except Exception as e:
            LOGGER.error(f"Benchmark run failed: {e}")

        return


if __name__ == "__main__":
    main()
