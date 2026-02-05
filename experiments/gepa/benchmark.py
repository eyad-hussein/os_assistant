"""Benchmarking utilities to compare assistant performance with and without GEPA prompt variants.

Provides a head-to-head runner that executes the `experiments/gepa/dataset.jsonl` entries
against a baseline (no variant) and one or more variants. Produces aggregated metrics
and saves a timestamped JSON result file under `experiments/gepa/results/`.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

from os_assistant.os_assistant import OSAssistant
from os_assistant.pydantic_models.schemas import FinalResult
from os_assistant.utils import LOGGER
from os_assistant.prompts.prompt_loader import list_prompt_variants


def _load_dataset(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _is_fallback_response(final_result: Optional[FinalResult]) -> bool:
    if not final_result:
        return True
    resp = final_result.response
    try:
        # Heuristic: fallback commands or info contain 'Unable to generate' or start with echo
        if final_result.response_type == "command":
            cmd = resp.command.lower()
            return cmd.startswith("echo") or "unable to generate" in cmd
        else:
            ans = resp.answer.lower()
            return "unable to generate" in ans or "i couldn't" in ans
    except Exception:
        return True


def _evaluate_example(record: Dict[str, Any], final_result: Optional[FinalResult]) -> Dict[str, Any]:
    expected_type = record.get("expected_type")
    expect_contains = record.get("expect_contains")
    expect_tool = record.get("expect_tool_needed", False)

    out = {
        "id": record.get("id"),
        "prompt": record.get("prompt"),
        "expected_type": expected_type,
        "got_type": final_result.response_type if final_result else None,
        "type_match": final_result is not None and final_result.response_type == expected_type,
        "parsing_fallback": _is_fallback_response(final_result),
        "tool_used": False,
        "expected_tool_needed": expect_tool,
        "command_contains_expected": None,
    }

    # Tool used detection
    if final_result and final_result.response:
        resp = final_result.response
        tool_exec = getattr(resp, "tool_execution", None)
        if tool_exec and (tool_exec.raw_output or tool_exec.code or tool_exec.analysis):
            out["tool_used"] = True

        if expected_type == "command" and expect_contains:
            try:
                cmd = resp.command or ""
                out["command_contains_expected"] = expect_contains.lower() in cmd.lower()
            except Exception:
                out["command_contains_expected"] = False

    return out


class BenchmarkRunner:
    def __init__(self, dataset_path: Path, results_dir: Path | None = None, limit: int | None = None):
        self.dataset_path = Path(dataset_path)
        self.results_dir = Path(results_dir or Path("experiments") / "gepa" / "results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.limit = limit

    def _run_once(self, variant: Optional[str], dataset: List[Dict[str, Any]], repeats: int = 1, seed: int | None = None) -> Dict[str, Any]:
        # Configure environment
        if variant:
            os.environ["PROMPT_VARIANT"] = variant
        elif "PROMPT_VARIANT" in os.environ:
            del os.environ["PROMPT_VARIANT"]

        runner_results = []
        for idx, rec in enumerate(dataset):
            if self.limit and idx >= self.limit:
                break

            # Run each example multiple times for statistical stability
            run_aggregates = []
            for r_i in range(repeats):
                # Optionally set deterministic seeds for repeatability
                if seed is not None:
                    try:
                        import random

                        random.seed(seed + r_i)
                    except Exception:
                        pass
                    os.environ["GEPA_RUN_SEED"] = str(seed + r_i)

                # Fresh assistant for each sample & repeat to avoid conversation bleed
                assistant = OSAssistant()
                try:
                    assistant.process_prompt(rec.get("prompt"))
                    state = assistant.app.get_state(config=assistant.config).values
                    final_result = state.get("final_result")
                except Exception as e:
                    LOGGER.exception(f"Error running prompt '{rec.get('id')}': {e}")
                    final_result = None

                evaluated = _evaluate_example(rec, final_result)
                run_aggregates.append(evaluated)

            # Aggregate across repeats (any-success policy for many metrics)
            agg = {
                "id": rec.get("id"),
                "prompt": rec.get("prompt"),
                "expected_type": rec.get("expected_type"),
                "got_type": None,
                "type_match": any(r.get("type_match") for r in run_aggregates),
                "parsing_fallback": all(r.get("parsing_fallback") for r in run_aggregates),
                "tool_used": any(r.get("tool_used") for r in run_aggregates),
                "expected_tool_needed": rec.get("expect_tool_needed", False),
                "command_contains_expected": any(r.get("command_contains_expected") for r in run_aggregates if r.get("command_contains_expected") is not None),
            }
            # choose a representative got_type (majority or first non-none)
            got_types = [r.get("got_type") for r in run_aggregates if r.get("got_type")]
            agg["got_type"] = got_types[0] if got_types else None

            runner_results.append(agg)

        # Aggregate
        total = len(runner_results)
        type_acc = mean(1 if r["type_match"] else 0 for r in runner_results) if total else 0
        parse_success = mean(1 if not r["parsing_fallback"] else 0 for r in runner_results) if total else 0
        tool_precision = None
        # Compute tool precision/recall if dataset contains expect_tool flags
        expected_tools = [r for r in runner_results if r.get("expected_tool_needed")]
        if expected_tools:
            true_pos = sum(1 for r in runner_results if r.get("tool_used") and r.get("expected_tool_needed"))
            predicted_pos = sum(1 for r in runner_results if r.get("tool_used"))
            tool_precision = true_pos / predicted_pos if predicted_pos else 0
        # command correctness
        command_checks = [r for r in runner_results if r.get("command_contains_expected") is not None]
        command_success = mean(1 if r.get("command_contains_expected") else 0 for r in command_checks) if command_checks else None

        summary = {
            "variant": variant,
            "num_examples": total,
            "type_accuracy": type_acc,
            "parsing_success": parse_success,
            "tool_precision": tool_precision,
            "command_success": command_success,
            "per_example": runner_results,
            "repeats": repeats,
            "seed": seed,
        }
        return summary

    def run_head_to_head(self, variants: List[str] | None = None, use_evaluator: bool = False, evaluator_dataset: str | None = None, batch_size: int = 1, repeats: int = 1, seed: int | None = None) -> Dict[str, Any]:
        dataset = _load_dataset(self.dataset_path)
        if self.limit:
            dataset = dataset[: self.limit]

        if not variants:
            # default: run against all variants discovered
            variants = [v.split(".")[1] for v in list_prompt_variants() if "." in v]

        results = {}

        if use_evaluator:
            # Use the OSAssistantEvaluator to run LLM-judged evaluations
            LOGGER.info("Running benchmark using OSAssistantEvaluator (LLM judge)...")
            from evaluator.core.evaluator import OSAssistantEvaluator

            evaluator = OSAssistantEvaluator(evaluator_dataset or str(self.dataset_path))
            evaluator.load_dataset(evaluator_dataset or str(self.dataset_path))

            # Baseline
            LOGGER.info("Running baseline (no GEPA variant) with LLM judge...")
            if "PROMPT_VARIANT" in os.environ:
                del os.environ["PROMPT_VARIANT"]
            baseline_results = []
            for idx, sample in enumerate(evaluator.dataset.samples):
                if self.limit and idx >= self.limit:
                    break
                # run repeats and average numeric scores where applicable
                scores = []
                for r_i in range(repeats):
                    if seed is not None:
                        os.environ["GEPA_RUN_SEED"] = str(seed + r_i)
                    scores.append(evaluator.evaluate_sample(sample.model_dump()))
                # average overall_score if present
                baseline_results.append(scores)

            results["baseline"] = {
                "num_examples": len(baseline_results),
                "per_example": baseline_results,
                "repeats": repeats,
                "seed": seed,
            }

            # Variants
            for v in variants:
                LOGGER.info(f"Running variant {v} with LLM judge...")
                os.environ["PROMPT_VARIANT"] = v
                variant_results = []
                for idx, sample in enumerate(evaluator.dataset.samples):
                    if self.limit and idx >= self.limit:
                        break
                    scores = []
                    for r_i in range(repeats):
                        if seed is not None:
                            os.environ["GEPA_RUN_SEED"] = str(seed + r_i)
                        scores.append(evaluator.evaluate_sample(sample.model_dump()))
                    variant_results.append(scores)
                results[v] = {
                    "num_examples": len(variant_results),
                    "per_example": variant_results,
                    "repeats": repeats,
                    "seed": seed,
                }

            # Save comparison
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            out_file = self.results_dir / f"benchmark_{ts}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, default=str)

            results["_results_path"] = str(out_file)
            LOGGER.info(f"Benchmark completed; saved to {out_file}")

            # Cleanup variant env
            if "PROMPT_VARIANT" in os.environ:
                del os.environ["PROMPT_VARIANT"]

            return results

        # Default non-LLM judge flow (fast heuristics)
        # Baseline (no variant)
        LOGGER.info("Running baseline (no GEPA variant)...")
        baseline_res = self._run_once(None, dataset)
        results["baseline"] = baseline_res

        for v in variants:
            LOGGER.info(f"Running variant: {v}")
            res = self._run_once(v, dataset)
            results[v] = res

        # Save comparison
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        out_file = self.results_dir / f"benchmark_{ts}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

        results["_results_path"] = str(out_file)
        LOGGER.info(f"Benchmark completed; saved to {out_file}")
        return results
