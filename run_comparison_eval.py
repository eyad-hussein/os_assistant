#!/usr/bin/env python3
"""
Comparison evaluation script for OS Assistant.

Runs evaluation with and without GEPA optimized prompts on GPT-4 mini.

Usage:
    python run_comparison_eval.py --dataset datasets/file_system_dataset.json --samples 20
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Ensure stdout uses UTF-8 to avoid console encoding errors on Windows
import sys
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Configuration
EVALUATOR_SCRIPT = "src/evaluator/run_comprehensive_eval.py"
RESULTS_DIR = Path("evaluation_results")


def run_evaluation(
    config_name: str,
    dataset_path: str,
    num_samples: int = 20,
    use_gepa_prompts: bool = False,
) -> dict:
    """
    Run evaluation with the specified configuration.

    Args:
        config_name: Name for this evaluation run (e.g., "baseline", "gepa-optimized")
        dataset_path: Path to the dataset file
        num_samples: Number of samples to evaluate
        use_gepa_prompts: Whether to use GEPA optimized prompts

    Returns:
        dict: Results of the evaluation
    """
    print(f"\n{'='*80}")
    print(f"Running evaluation: {config_name}")
    print(f"{'='*80}")
    print(f"Dataset: {dataset_path}")
    print(f"Samples to evaluate: {num_samples}")
    print(f"GEPA optimized prompts: {use_gepa_prompts}")
    print(f"Model: {os.environ.get('MODEL_JUDGE_NAME', 'gpt-4-mini')}")

    # Set up environment variables
    env = os.environ.copy()
    env["MODEL_JUDGE_NAME"] = os.environ.get("MODEL_JUDGE_NAME", "gpt-4-mini")

    if use_gepa_prompts:
        env["USE_GEPA_PROMPTS"] = "1"
        env["PROMPT_VARIANT"] = "gepa-optimized"
    else:
        env["USE_GEPA_PROMPTS"] = "0"
        env["PROMPT_VARIANT"] = "default"

    # Prepare output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    variant = "gepa" if use_gepa_prompts else "baseline"
    output_file = RESULTS_DIR / f"evaluation_{config_name}_{variant}_{timestamp}.json"

    # Build command
    cmd = [
        sys.executable,
        EVALUATOR_SCRIPT,
        "--dataset",
        dataset_path,
        "--batch-size",
        "5",
        "--output",
        str(output_file),
        "--end",
        str(max(0, num_samples - 1)),  # end is inclusive, so subtract 1, ensure at least one sample is evaluated
        "--verbose",
    ]

    print(f"\nCommand: {' '.join(cmd)}")
    print(f"Output file: {output_file}")
    print("\nStarting evaluation...")

    try:
        result = subprocess.run(
            cmd,
            env=env,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        print(result.stdout)

        # Try to parse and return results
        if output_file.exists():
            with open(output_file, "r") as f:
                eval_results = json.load(f)
            return {
                "config": config_name,
                "variant": variant,
                "output_file": str(output_file),
                "status": "completed",
                "results": eval_results,
            }
        else:
                # Fallback: try to find any recent evaluator output for this dataset
                dataset_base = Path(dataset_path).stem
                candidates = sorted(
                    RESULTS_DIR.glob(f"{dataset_base}_eval*.json"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if candidates:
                    fallback_file = candidates[0]
                    try:
                        with open(fallback_file, "r", encoding="utf-8") as f:
                            eval_results = json.load(f)
                        return {
                            "config": config_name,
                            "variant": variant,
                            "output_file": str(fallback_file),
                            "status": "completed",
                            "results": eval_results,
                        }
                    except Exception:
                        pass

                return {
                    "config": config_name,
                    "variant": variant,
                    "status": "completed",
                    "message": "Evaluation completed but could not find output file",
                }

    except subprocess.CalledProcessError as e:
        print(f"\nError during evaluation: {e.returncode}")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        return {
            "config": config_name,
            "variant": variant,
            "status": "failed",
            "error": str(e),
            "stderr": e.stderr,
        }


def compare_results(baseline_results: dict, gepa_results: dict) -> None:
    """
    Compare evaluation results between baseline and GEPA optimized runs.

    Args:
        baseline_results: Results from baseline evaluation
        gepa_results: Results from GEPA optimized evaluation
    """
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")

    if baseline_results.get("status") != "completed" or gepa_results.get("status") != "completed":
        print("Cannot compare results - one or both evaluations failed")
        return

    base_res = baseline_results.get("results", {})
    gepa_res = gepa_results.get("results", {})

    # Extract summary metrics if available
    base_metrics = base_res.get("summary", {})
    gepa_metrics = gepa_res.get("summary", {})

    print("\nMetric Comparison:")
    print(f"{'Metric':<30} {'Baseline':<20} {'GEPA':<20} {'Improvement':<15}")
    print("-" * 85)

    # Summary keys from evaluator.generate_summary():
    # average_score, average_correctness, average_completeness, average_clarity
    metrics_to_compare = [
        ("average_score", "Overall Score"),
        ("average_correctness", "Correctness"),
        ("average_completeness", "Completeness"),
        ("average_clarity", "Clarity"),
    ]

    for metric_key, metric_name in metrics_to_compare:
        baseline_val = base_metrics.get(metric_key, 0)
        gepa_val = gepa_metrics.get(metric_key, 0)

        if baseline_val and gepa_val:
            improvement = gepa_val - baseline_val
            improvement_pct = (improvement / baseline_val * 100) if baseline_val > 0 else 0
            improvement_str = f"{improvement:+.2f} ({improvement_pct:+.1f}%)"
        else:
            improvement_str = "N/A"

        print(f"{metric_name:<30} {baseline_val:<20.2f} {gepa_val:<20.2f} {improvement_str:<15}")

    # Show detailed results files
    print(f"\nDetailed Results Files:")
    print(f"- Baseline: {baseline_results.get('output_file', 'N/A')}")
    print(f"- GEPA:     {gepa_results.get('output_file', 'N/A')}")


def main():
    """Main entry point for the comparison evaluation."""
    parser = argparse.ArgumentParser(
        description="Run OS Assistant evaluation comparison with and without GEPA optimized prompts"
    )

    parser.add_argument(
        "--dataset",
        default="datasets/file_system_dataset.json",
        help="Path to the evaluation dataset",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of samples to evaluate",
    )

    parser.add_argument(
        "--model",
        default="gpt-4-mini",
        help="Model to use for evaluation (default: gpt-4-mini)",
    )

    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline evaluation (only run GEPA)",
    )

    parser.add_argument(
        "--skip-gepa",
        action="store_true",
        help="Skip GEPA evaluation (only run baseline)",
    )

    args = parser.parse_args()

    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Set model in environment
    os.environ["MODEL_JUDGE_NAME"] = args.model

    print("\n" + "=" * 80)
    print("OS ASSISTANT EVALUATION COMPARISON")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Samples: {args.samples}")

    results = {}

    # Run baseline evaluation
    if not args.skip_baseline:
        print("\n[1/2] Running baseline evaluation (without GEPA optimized prompts)...")
        results["baseline"] = run_evaluation(
            "comparison",
            args.dataset,
            args.samples,
            use_gepa_prompts=False,
        )
    else:
        print("\n[1/2] Skipping baseline evaluation")

    # Run GEPA evaluation
    if not args.skip_gepa:
        print("\n[2/2] Running GEPA optimized evaluation...")
        results["gepa"] = run_evaluation(
            "comparison",
            args.dataset,
            args.samples,
            use_gepa_prompts=True,
        )
    else:
        print("\n[2/2] Skipping GEPA evaluation")

    # Compare results
    if "baseline" in results and "gepa" in results:
        compare_results(results["baseline"], results["gepa"])
    else:
        print("\nCannot perform comparison - not all evaluations were run")

    # Save comparison metadata
    comparison_file = RESULTS_DIR / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(comparison_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nComparison metadata saved to: {comparison_file}")

    print("\n" + "=" * 80)
    print("Evaluation comparison complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
