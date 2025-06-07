import argparse
from datetime import datetime

from evaluator.core.evaluator import OSAssistantEvaluator


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run OS Assistant evaluation")

    parser.add_argument(
        "--dataset",
        default="datasets/file_system_dataset.json",
        help="Path to the evaluation dataset",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of samples to evaluate before saving results",
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

    return parser.parse_args()


def main():
    """Run the evaluation process with specified parameters."""
    # Parse command line arguments
    args = parse_arguments()

    # Initialize evaluator
    evaluator = OSAssistantEvaluator(args.dataset)

    # Load dataset
    dataset = evaluator.load_dataset()
    print(f"Dataset loaded: {dataset.metadata.get('description', 'No description')}")
    print(f"Total samples in dataset: {len(dataset.samples)}")

    # Convert end index if specified
    end_index = int(args.end) if args.end is not None else None

    # Run evaluation with specified parameters
    print(f"\nStarting evaluation with query range [{args.start}:{args.end or 'end'}]")
    print(f"Batch size: {args.batch_size}")
    if args.continue_from:
        print(f"Continuing from: {args.continue_from}")

    start_time = datetime.now()
    results = evaluator.run_evaluation(
        start_index=args.start,
        end_index=end_index,
        batch_size=args.batch_size,
        continue_from=args.continue_from,
    )
    end_time = datetime.now()

    total_duration = (end_time - start_time).total_seconds()

    # Generate final summary
    summary = evaluator.generate_summary()

    # Print summary information
    print("\n===== FINAL EVALUATION SUMMARY =====")
    print(f"Total samples evaluated: {summary.total_samples}")
    print(f"Overall average score: {summary.average_score:.2f}/5.0")
    print(f"Average correctness: {summary.average_correctness:.2f}/5.0")
    print(f"Average completeness: {summary.average_completeness:.2f}/5.0")

    print("\nScores by domain:")
    for domain, score in summary.domain_scores.items():
        print(f"  - {domain}: {score:.2f}/5.0")

    print("\nLatency Metrics:")
    print(
        f"  - Average prompt processing: {summary.latency_metrics['avg_prompt_processing_ms']:.2f} ms"
    )
    print(
        f"  - Average LLM evaluation: {summary.latency_metrics['avg_llm_evaluation_ms']:.2f} ms"
    )
    print(
        f"  - Average total per sample: {summary.latency_metrics['avg_total_evaluation_ms']:.2f} ms"
    )
    print(f"  - Fastest sample: {summary.latency_metrics['min_total_ms']:.2f} ms")
    print(f"  - Slowest sample: {summary.latency_metrics['max_total_ms']:.2f} ms")
    print(f"  - Total evaluation time: {total_duration:.2f} seconds")

    # Update the results file with final summary
    evaluator.save_results()
    print(f"\nFinal evaluation complete")


if __name__ == "__main__":
    main()
