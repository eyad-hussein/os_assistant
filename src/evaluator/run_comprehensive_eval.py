import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List

from evaluator.config.config import (
    BATCH_SIZE,
    DATASETS_DIR,
    DETAILED_REPORTS,
    EVALUATION_METRICS,
    RESULTS_DIR,
    THRESHOLDS,
)
from evaluator.core.evaluator import OSAssistantEvaluator


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive OS Assistant evaluation with improved metrics"
    )

    parser.add_argument(
        "--dataset",
        default=os.path.join(DATASETS_DIR, "file_system_dataset.json"),
        help="Path to the evaluation dataset",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Number of samples to evaluate before saving results (default: {BATCH_SIZE})",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Custom output filename",
    )

    parser.add_argument(
        "--start", type=int, default=0, help="Index of the first sample to evaluate"
    )

    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Index of the last sample to evaluate (None for all)",
    )

    parser.add_argument(
        "--continue-from",
        default=None,
        help="Path to existing evaluation file to continue from",
    )

    parser.add_argument(
        "--verbose", action="store_true", help="Display detailed evaluation progress"
    )

    return parser.parse_args()


def print_detailed_table(summary):
    """Print a detailed table of results without using tabulate."""
    if not DETAILED_REPORTS:
        return

    # Print header
    headers = ["Domain", "Type", "Correctness", "Completeness", "Clarity", "Overall"]
    header_str = " | ".join(headers)
    separator = "-" * len(header_str)

    print("\nDetailed Results:")
    print(separator)
    print(header_str)
    print(separator)

    # Print rows
    for result in summary.get("detailed_results", []):
        row = [
            result.get("domain", "unknown"),
            result.get("type", "unknown"),
            f"{result.get('correctness', 0):.1f}",
            f"{result.get('completeness', 0):.1f}",
            f"{result.get('clarity', 0):.1f}",
            f"{result.get('overall_score', 0):.1f}",
        ]
        print(" | ".join(row))

    print(separator)


def main():
    """Run the comprehensive evaluation process with improved metrics."""
    # Parse command line arguments
    args = parse_arguments()

    # Set up output path
    if args.output:
        output_path = args.output
    else:
        output_dir = RESULTS_DIR
        os.makedirs(output_dir, exist_ok=True)
        dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
        output_path = os.path.join(output_dir, f"{dataset_name}_eval.json")

    # Initialize evaluator
    batch_size = args.batch_size
    evaluator = OSAssistantEvaluator(args.dataset)

    # Print evaluation setup
    print("=" * 80)
    print("OS ASSISTANT LIVE EVALUATION")
    print("=" * 80)
    print(
        "This evaluation uses the ACTUAL OS Assistant to generate responses in real-time!"
    )
    print(f"Dataset: {args.dataset}")
    print(f"Output: {output_path}")
    print(f"Batch size: {batch_size}")
    if args.continue_from:
        print(f"Continuing from: {args.continue_from}")

    # Configure metrics from config
    metrics_str = ", ".join(
        [f"{m['name']} ({m['weight']*100:.0f}%)" for m in EVALUATION_METRICS]
    )
    print(f"Evaluation metrics: {metrics_str}")

    # Load dataset
    dataset = evaluator.load_dataset()
    print(f"Dataset loaded: {dataset.metadata.get('description', 'No description')}")
    print(f"Total samples in dataset: {len(dataset.samples)}")

    # Determine sample range
    end_index = int(args.end) if args.end is not None else None
    samples_to_process = len(dataset.samples[args.start : end_index])
    print(
        f"\nWill evaluate {samples_to_process} samples (index range: {args.start} to {end_index or 'end'})"
    )

    # Start evaluation
    print("\nStarting live evaluation using actual OS Assistant...")
    start_time = datetime.now()

    try:
        # Run evaluation
        results = evaluator.run_evaluation(
            start_index=args.start,
            end_index=end_index,
            batch_size=batch_size,
            continue_from=args.continue_from,
        )

        # Generate final summary
        summary = evaluator.generate_summary()

        # Show final summary
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()

        print("\n" + "=" * 80)
        print("EVALUATION RESULTS SUMMARY")
        print("=" * 80)
        print(f"Total samples evaluated: {summary.total_samples}")
        print(f"Overall average score: {summary.average_score:.2f}/5.0")

        # Print metric averages
        print("\nMetric Averages:")
        print(f"- Correctness:  {summary.average_correctness:.2f}/5.0")
        print(f"- Completeness: {summary.average_completeness:.2f}/5.0")
        print(f"- Clarity:      {summary.average_clarity:.2f}/5.0")

        # Print domain scores
        print("\nScores by Domain:")
        for domain, score in summary.domain_scores.items():
            print(f"- {domain}: {score:.2f}/5.0")

        # Evaluate against thresholds
        print("\nPerformance Assessment:")
        overall_assessment = "Poor"
        if summary.average_score >= THRESHOLDS["excellent"]:
            overall_assessment = "Excellent"
        elif summary.average_score >= THRESHOLDS["good"]:
            overall_assessment = "Good"
        elif summary.average_score >= THRESHOLDS["acceptable"]:
            overall_assessment = "Acceptable"
        print(
            f"Overall Performance: {overall_assessment} ({summary.average_score:.2f}/5.0)"
        )
        # Print latency metrics
        print("\nLatency Metrics:")
        print(
            f"- Average evaluation time per sample: {summary.latency_metrics['avg_total_evaluation_ms']/1000:.2f} seconds"
        )
        print(f"- Total evaluation time: {total_duration:.2f} seconds")

        # Print detailed table if requested
        if args.verbose or DETAILED_REPORTS:
            print_detailed_table(summary.model_dump())

        print(f"\nEvaluation complete! Results saved to {output_path}")

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user. Partial results saved.")
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
