from datetime import datetime

from evaluator.core.evaluator import OSAssistantEvaluator


def main():
    """Run the evaluation process."""
    # Set dataset path
    dataset_path = "datasets/example.json"

    # Initialize evaluator
    evaluator = OSAssistantEvaluator(dataset_path)

    # Load dataset
    dataset = evaluator.load_dataset()
    print(f"Dataset loaded: {dataset.metadata.get('description', 'No description')}")
    print(f"Number of samples: {len(dataset.samples)}")

    # Run evaluation
    print("\nStarting evaluation...")
    results = evaluator.run_evaluation()

    # Generate summary
    summary = evaluator.generate_summary()

    # Print summary information
    print("\n===== EVALUATION SUMMARY =====")
    print(f"Total samples evaluated: {summary.total_samples}")
    print(f"Overall average score: {summary.average_score:.2f}/5.0")

    if summary.command_average:
        print(f"Command queries average: {summary.command_average:.2f}/5.0")

    if summary.information_average:
        print(f"Information queries average: {summary.information_average:.2f}/5.0")

    print("\nScores by domain:")
    for domain, score in summary.domain_scores.items():
        print(f"  - {domain}: {score:.2f}/5.0")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"evaluation_results/evaluation_{timestamp}.json"
    evaluator.save_results(output_path)


if __name__ == "__main__":
    main()
