import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from os_assistant.os_assistant import OSAssistant

from ..utils.models import EvaluationDataset, EvaluationSummary, LatencyMetrics
from ..utils.parser import extract_final_result
from .llm_judge import LLMJudge


class OSAssistantEvaluator:
    """Evaluator for OS Assistant performance."""

    def __init__(self, dataset_path: str | None = None):
        """Initialize the evaluator.

        Args:
            dataset_path: Path to the evaluation dataset JSON file
        """
        self.dataset_path = dataset_path
        self.dataset = None
        self.results = []
        self.judge = LLMJudge()
        self.output_file_path = None

    def load_dataset(self, dataset_path: str | None = None) -> EvaluationDataset:
        """Load the evaluation dataset.

        Args:
            dataset_path: Path to the dataset JSON file (optional override)

        Returns:
            The loaded dataset
        """
        if dataset_path:
            self.dataset_path = dataset_path

        if not self.dataset_path:
            raise ValueError("Dataset path not provided")

        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            dataset_dict = json.load(f)

        # Validate using Pydantic
        self.dataset = EvaluationDataset(**dataset_dict)
        print(f"Loaded dataset with {len(self.dataset.samples)} samples")

        return self.dataset

    def evaluate_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single sample from the dataset.

        Args:
            sample: A single sample from the dataset

        Returns:
            Evaluation results for the sample
        """
        question = sample["question"]
        expected_response = sample["expected_response"]
        query_type = sample["type"]
        domain = sample["domain"]

        print(f"\nEvaluating question: {question}")
        print(f"Type: {query_type}, Domain: {domain}")

        # Initialize OS Assistant
        assistant = OSAssistant()

        # Process the question with latency tracking
        start_time = time.time()
        assistant.process_prompt(question)
        prompt_processing_time = time.time() - start_time
        prompt_processing_ms = prompt_processing_time * 1000

        # Extract final result from assistant state
        state = assistant.app.get_state(config=assistant.config).values
        final_result = extract_final_result(state)

        if not final_result:
            print("Warning: No final result found in assistant state")
            return {
                "sample": sample,
                "actual_response": None,
                "evaluation": {
                    "scores": {
                        "correctness": 0,
                        "completeness": 0,
                    },
                    "correctness_explanation": "Assistant did not generate a final result",
                    "completeness_explanation": "Assistant did not generate a final result",
                    "overall_score": 0.0,
                    "reasoning": "Assistant did not generate a final result",
                },
                "latency": {
                    "prompt_processing_ms": prompt_processing_ms,
                    "llm_evaluation_ms": 0,
                    "total_evaluation_ms": prompt_processing_ms,
                },
            }

        # Get the actual response
        actual_response = final_result.get("response", {})
        actual_response_type = final_result.get("response_type", query_type)

        # Evaluate using LLM Judge
        evaluation, llm_evaluation_ms = self.judge.evaluate(
            question=question,
            expected_response=expected_response,
            actual_response=actual_response,
            query_type=query_type,
        )

        # Calculate total evaluation time
        total_evaluation_ms = prompt_processing_ms + llm_evaluation_ms

        # Create latency metrics
        latency = {
            "prompt_processing_ms": prompt_processing_ms,
            "llm_evaluation_ms": llm_evaluation_ms,
            "total_evaluation_ms": total_evaluation_ms,
        }

        # Create result record
        result = {
            "sample": sample,
            "actual_response": actual_response,
            "actual_response_type": actual_response_type,
            "evaluation": evaluation,
            "latency": latency,
        }

        # Print the evaluation details including explanations
        self._print_evaluation_details(result)

        return result

    def _print_evaluation_details(self, result: Dict[str, Any]) -> None:
        """Print detailed evaluation results for a sample.

        Args:
            result: The evaluation result dictionary
        """
        evaluation = result.get("evaluation", {})
        scores = evaluation.get("scores", {})

        correctness = scores.get("correctness", 0)
        completeness = scores.get("completeness", 0)
        overall = evaluation.get("overall_score", 0)

        print("\n=== EVALUATION RESULTS ===")
        print(f"Correctness: {correctness}/5")
        print(
            f"Explanation: {evaluation.get('correctness_explanation', 'No explanation provided')}"
        )

        print(f"\nCompleteness: {completeness}/5")
        print(
            f"Explanation: {evaluation.get('completeness_explanation', 'No explanation provided')}"
        )

        print(f"\nOverall Score: {overall:.1f}/5.0")
        print(f"Reasoning: {evaluation.get('reasoning', 'No reasoning provided')}")

        print(
            f"\nLatency: Processing={result.get('latency', {}).get('prompt_processing_ms', 0):.1f}ms, "
            f"Evaluation={result.get('latency', {}).get('llm_evaluation_ms', 0):.1f}ms"
        )
        print("=" * 50)

    def run_evaluation(
        self,
        start_index: int = 0,
        end_index: Optional[int] = None,
        batch_size: int = 5,
        continue_from: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run evaluation on the dataset with incremental saving to a single file.

        Args:
            start_index: Index of the first sample to evaluate (default: 0)
            end_index: Index of the last sample to evaluate (default: None, evaluate till end)
            batch_size: Number of samples to process before updating results (default: 5)
            continue_from: Path to existing evaluation file to continue from (default: None)

        Returns:
            List of evaluation results
        """
        if not self.dataset:
            self.load_dataset()

        samples = self.dataset.samples

        # Validate indices
        if start_index < 0:
            start_index = 0
        if end_index is None or end_index >= len(samples):
            end_index = len(samples) - 1
        if start_index > end_index:
            raise ValueError(
                f"Start index {start_index} cannot be greater than end index {end_index}"
            )

        # Select sample range to evaluate
        samples_to_evaluate = samples[start_index : end_index + 1]

        print(
            f"Will evaluate samples from index {start_index} to {end_index} (total: {len(samples_to_evaluate)})"
        )

        # Handle continuing from existing evaluation
        if continue_from and os.path.exists(continue_from):
            self._continue_evaluation(continue_from)
            print(
                f"Continuing evaluation from {continue_from} with {len(self.results)} existing results"
            )
        else:
            self.results = []

            # Create the output file path with timestamp to make it unique for this run
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_file_path = f"evaluation_results/evaluation_{timestamp}.json"

            # Initialize the results file with a skeleton structure
            self._initialize_results_file()

        batch_count = len(self.results) // batch_size

        # Track already evaluated samples
        evaluated_indices = set([r.get("sample_index", -1) for r in self.results])

        for i, sample in enumerate(samples_to_evaluate):
            absolute_index = i + start_index

            # Skip already evaluated samples
            if absolute_index in evaluated_indices:
                print(f"\nSkipping already evaluated sample {absolute_index}")
                continue

            print(
                f"\nEvaluating sample {absolute_index} ({i+1}/{len(samples_to_evaluate)})"
            )
            sample_dict = sample.model_dump()
            result = self.evaluate_sample(sample_dict)

            # Store the sample index for reference
            result["sample_index"] = absolute_index

            self.results.append(result)

            # Update results file after each batch
            if (len(self.results) % batch_size == 0) or (
                i == len(samples_to_evaluate) - 1
            ):
                batch_count += 1
                print(f"\nCompleted batch {batch_count}. Updating results file...")

                # Calculate and display metrics for this batch
                last_batch_results = self.results[-min(batch_size, len(self.results)) :]
                batch_metrics = self._generate_batch_summary(last_batch_results)

                print("\n===== CURRENT BATCH METRICS =====")
                print(f"Batch size: {len(last_batch_results)}")
                print(f"Average score: {batch_metrics['average_score']:.2f}/5.0")
                print(
                    f"Average correctness: {batch_metrics['average_correctness']:.2f}/5.0"
                )
                print(
                    f"Average completeness: {batch_metrics['average_completeness']:.2f}/5.0"
                )

                # Calculate running averages for all results so far
                all_metrics = self._generate_batch_summary(self.results)

                print("\n===== RUNNING AVERAGE METRICS =====")
                print(f"Total evaluated: {len(self.results)}/{len(samples)}")
                print(f"Average score: {all_metrics['average_score']:.2f}/5.0")
                print(
                    f"Average correctness: {all_metrics['average_correctness']:.2f}/5.0"
                )
                print(
                    f"Average completeness: {all_metrics['average_completeness']:.2f}/5.0"
                )

                # Update the results file with this batch
                self._update_results_file(last_batch_results, batch_count)

        print(f"\nAll evaluations completed and saved to {self.output_file_path}")
        return self.results

    def _continue_evaluation(self, file_path: str) -> None:
        """Load existing evaluation results to continue from.

        Args:
            file_path: Path to the existing evaluation file
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Set the output file path to continue using the same file
        self.output_file_path = file_path

        # Extract all results from previous batches
        self.results = []
        for batch in data.get("batches", []):
            self.results.extend(batch.get("results", []))

        print(f"Loaded {len(self.results)} existing evaluation results")

    def _initialize_results_file(self) -> None:
        """Initialize the results file with a skeleton structure."""
        # Create initial structure
        initial_data = {
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "dataset": self.dataset_path,
                "start_time": datetime.now().isoformat(),
            },
            "batches": [],
            "summary": None,  # Will be populated at the end
        }

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.output_file_path), exist_ok=True)

        # Write initial structure to file
        with open(self.output_file_path, "w", encoding="utf-8") as f:
            json.dump(initial_data, f, indent=2)

        print(f"Initialized results file at {self.output_file_path}")

    def _update_results_file(
        self, batch_results: List[Dict[str, Any]], batch_num: int
    ) -> None:
        """Update the results file with a new batch of results.

        Args:
            batch_results: The results for this batch
            batch_num: The batch number
        """
        try:
            # Read current file content
            with open(self.output_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # If file doesn't exist or is corrupt, initialize it
            self._initialize_results_file()
            with open(self.output_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

        # Generate batch summary
        batch_summary = self._generate_batch_summary(batch_results)

        # Add timestamp to batch summary
        batch_summary["timestamp"] = datetime.now().isoformat()

        # Create batch entry
        batch_entry = {
            "batch_number": batch_num,
            "timestamp": datetime.now().isoformat(),
            "batch_summary": batch_summary,
            "results": batch_results,
        }

        # Add this batch to the batches array
        data["batches"].append(batch_entry)

        # Update the running metrics
        all_results = []
        for batch in data.get("batches", []):
            all_results.extend(batch.get("results", []))

        running_metrics = self._generate_batch_summary(all_results)
        data["running_metrics"] = running_metrics

        # Write updated data back to file
        with open(self.output_file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(f"Updated results file with batch {batch_num}")

    def _generate_batch_summary(
        self, batch_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate a summary for a specific batch of results.

        Args:
            batch_results: The results for this batch

        Returns:
            Summary statistics for the batch
        """
        if not batch_results:
            return {}

        total_samples = len(batch_results)

        # Extract scores from this batch
        all_scores = [
            r.get("evaluation", {}).get("overall_score", 0) for r in batch_results
        ]
        all_correctness = [
            r.get("evaluation", {}).get("scores", {}).get("correctness", 0)
            for r in batch_results
        ]
        all_completeness = [
            r.get("evaluation", {}).get("scores", {}).get("completeness", 0)
            for r in batch_results
        ]

        # Calculate averages for this batch
        avg_score = sum(all_scores) / total_samples if total_samples > 0 else 0
        avg_correctness = (
            sum(all_correctness) / total_samples if total_samples > 0 else 0
        )
        avg_completeness = (
            sum(all_completeness) / total_samples if total_samples > 0 else 0
        )

        # Return batch summary
        return {
            "batch_size": total_samples,
            "average_score": avg_score,
            "average_correctness": avg_correctness,
            "average_completeness": avg_completeness,
        }

    def generate_summary(self) -> EvaluationSummary:
        """Generate a summary of the evaluation results.

        Returns:
            Summary statistics
        """
        if not self.results:
            raise ValueError("No evaluation results to summarize")

        total_samples = len(self.results)

        # Extract all scores
        all_scores = [
            r.get("evaluation", {}).get("overall_score", 0) for r in self.results
        ]
        all_correctness = [
            r.get("evaluation", {}).get("scores", {}).get("correctness", 0)
            for r in self.results
        ]
        all_completeness = [
            r.get("evaluation", {}).get("scores", {}).get("completeness", 0)
            for r in self.results
        ]

        # Calculate averages
        avg_score = sum(all_scores) / total_samples if total_samples > 0 else 0
        avg_correctness = (
            sum(all_correctness) / total_samples if total_samples > 0 else 0
        )
        avg_completeness = (
            sum(all_completeness) / total_samples if total_samples > 0 else 0
        )

        # Calculate latency metrics
        all_prompt_latencies = [
            r.get("latency", {}).get("prompt_processing_ms", 0) for r in self.results
        ]
        all_llm_latencies = [
            r.get("latency", {}).get("llm_evaluation_ms", 0) for r in self.results
        ]
        all_total_latencies = [
            r.get("latency", {}).get("total_evaluation_ms", 0) for r in self.results
        ]

        avg_prompt_latency = (
            sum(all_prompt_latencies) / total_samples if total_samples > 0 else 0
        )
        avg_llm_latency = (
            sum(all_llm_latencies) / total_samples if total_samples > 0 else 0
        )
        avg_total_latency = (
            sum(all_total_latencies) / total_samples if total_samples > 0 else 0
        )

        # Group scores by type
        command_scores = []
        info_scores = []
        for result in self.results:
            if result["sample"]["type"] == "command":
                command_scores.append(
                    result.get("evaluation", {}).get("overall_score", 0)
                )
            else:
                info_scores.append(result.get("evaluation", {}).get("overall_score", 0))

        command_avg = (
            sum(command_scores) / len(command_scores) if command_scores else None
        )
        info_avg = sum(info_scores) / len(info_scores) if info_scores else None

        # Group scores by domain
        domain_scores = {}
        for result in self.results:
            domain = result["sample"]["domain"]
            if domain not in domain_scores:
                domain_scores[domain] = {"scores": [], "avg": 0}

            domain_scores[domain]["scores"].append(
                result.get("evaluation", {}).get("overall_score", 0)
            )

        # Calculate domain averages
        for domain, data in domain_scores.items():
            scores = data["scores"]
            data["avg"] = sum(scores) / len(scores) if scores else 0

        # Format domain scores for the summary
        domain_avgs = {domain: data["avg"] for domain, data in domain_scores.items()}

        # Create detailed results for the summary
        detailed = []
        for result in self.results:
            detailed.append(
                {
                    "question": result["sample"]["question"],
                    "type": result["sample"]["type"],
                    "domain": result["sample"]["domain"],
                    "correctness": result.get("evaluation", {})
                    .get("scores", {})
                    .get("correctness", 0),
                    "completeness": result.get("evaluation", {})
                    .get("scores", {})
                    .get("completeness", 0),
                    "overall_score": result.get("evaluation", {}).get(
                        "overall_score", 0
                    ),
                    "latency_ms": result.get("latency", {}).get(
                        "total_evaluation_ms", 0
                    ),
                    "reasoning": result.get("evaluation", {}).get("reasoning", ""),
                }
            )

        # Create the summary
        summary = EvaluationSummary(
            total_samples=total_samples,
            average_score=avg_score,
            average_correctness=avg_correctness,
            average_completeness=avg_completeness,
            command_average=command_avg,
            information_average=info_avg,
            domain_scores=domain_avgs,
            latency_metrics={
                "avg_prompt_processing_ms": avg_prompt_latency,
                "avg_llm_evaluation_ms": avg_llm_latency,
                "avg_total_evaluation_ms": avg_total_latency,
                "min_total_ms": min(all_total_latencies) if all_total_latencies else 0,
                "max_total_ms": max(all_total_latencies) if all_total_latencies else 0,
            },
            detailed_results=detailed,
        )

        return summary

    def save_results(self, output_path: str | None = None) -> None:
        """Save the final evaluation results or update the existing file.

        Args:
            output_path: Optional custom path to save the results. If None, uses the incremental file.
        """
        if not self.results:
            raise ValueError("No evaluation results to save")

        # Use the incremental file if no custom path provided
        final_path = output_path or self.output_file_path

        if final_path == self.output_file_path:
            # If using the incremental file, just update the summary
            with open(self.output_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Generate and add the final summary
            data["summary"] = self.generate_summary().model_dump()
            data["metadata"]["end_time"] = datetime.now().isoformat()

            # Write updated data back to file
            with open(self.output_file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            print(f"Updated final summary in {self.output_file_path}")
        else:
            # Otherwise create a new complete file
            # Generate summary
            summary = self.generate_summary()

            # Create full report
            report = {
                "timestamp": datetime.now().isoformat(),
                "summary": summary.model_dump(),
                "detailed_results": self.results,
            }

            # Ensure directory exists
            os.makedirs(os.path.dirname(final_path), exist_ok=True)

            # Save to file
            with open(final_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

            print(f"Saved complete evaluation results to {final_path}")
