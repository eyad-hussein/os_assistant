import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from os_assistant.os_assistant import OSAssistant

from evaluator.config.config import RESULTS_DIR
from evaluator.core.llm_judge import LLMJudge
from evaluator.utils.models import (
    DatasetSample,
    EvaluationDataset,
    EvaluationSummary,
    RunningMetrics,
)
from evaluator.utils.parser import extract_final_result, validate_evaluation_result


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

    def _convert_to_serializable(self, obj):
        """Convert a complex object to a JSON serializable form.

        Args:
            obj: The object to convert

        Returns:
            A JSON serializable representation of the object
        """
        if obj is None:
            return None

        # Handle pydantic models
        if hasattr(obj, "model_dump"):
            return self._convert_to_serializable(obj.model_dump())
        elif hasattr(obj, "dict"):
            return self._convert_to_serializable(obj.dict())

        # Handle dictionaries
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}

        # Handle lists and tuples
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]

        # Handle sets
        elif isinstance(obj, set):
            return [self._convert_to_serializable(item) for item in obj]

        # Handle datetime objects
        elif hasattr(obj, "isoformat"):
            return obj.isoformat()

        # For any other objects that might not be serializable, convert to string
        try:
            # First try json serialization to test if it's already serializable
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError, ValueError):
            # If serialization fails, convert to string
            return str(obj)

    def evaluate_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single sample from the dataset using the actual OS Assistant.

        Args:
            sample: A single sample from the dataset

        Returns:
            Evaluation results for the sample
        """
        # Extract only what we need from the sample
        question = sample["question"]
        expected_response = sample["expected_response"]
        query_type = sample["type"]
        domain = sample["domain"]

        # Print ONLY the question without type and domain information
        print(f"\nEvaluating: {question}")

        # Initialize OS Assistant - this is the real OSAssistant that will generate a response
        assistant = OSAssistant()

        # Process the question with latency tracking - only pass the question itself
        start_time = time.time()
        assistant.process_prompt(question)  # Pass the question to the assistant
        prompt_processing_time = time.time() - start_time
        prompt_processing_ms = prompt_processing_time * 1000

        # Extract final result from assistant state - this is the actual generated response
        state = assistant.app.get_state(config=assistant.config).values
        final_result = extract_final_result(state)

        if not final_result:
            print("Warning: No final result found in assistant state")
            return {
                "sample": sample,
                "actual_response": None,
                "expected_response": expected_response,
                "evaluation": {
                    "scores": {
                        "correctness": 0,
                        "completeness": 0,
                        "clarity": 0,
                    },
                    "correctness_explanation": "Assistant did not generate a final result",
                    "completeness_explanation": "Assistant did not generate a final result",
                    "clarity_explanation": "Assistant did not generate a final result",
                    "overall_score": 0.0,
                    "reasoning": "Assistant did not generate a final result",
                },
                "latency": {
                    "prompt_processing_ms": prompt_processing_ms,
                    "llm_evaluation_ms": 0,
                    "total_evaluation_ms": prompt_processing_ms,
                },
            }

        # Get the actual response - handle both dict and Pydantic model cases
        actual_response = self._get_attribute_safely(final_result, "response", {})
        actual_response_type = self._get_attribute_safely(
            final_result, "response_type", query_type
        )

        # Evaluate using LLM Judge - compare actual response against expected response
        try:
            evaluation_result, latency_metrics = self.judge.evaluate(
                question=question,
                expected_response=expected_response,
                actual_response=actual_response,  # Pass the actual generated response
                query_type=query_type,
            )

            # Extract the total evaluation time
            total_evaluation_ms = prompt_processing_ms + latency_metrics.get(
                "total_evaluation_ms", 0
            )

            # Update the latency metrics
            latency_metrics["prompt_processing_ms"] = prompt_processing_ms
            latency_metrics["total_evaluation_ms"] = total_evaluation_ms

            # Get evaluation scores and prepare result
            validated_result = validate_evaluation_result(evaluation_result)

            # Extract scores from the validated result
            scores = validated_result.get("scores", {})
            correctness = scores.get("correctness", 3.0)  # Default to 3.0 if missing
            completeness = scores.get("completeness", 3.0)  # Default to 3.0 if missing
            clarity = scores.get("clarity", 3.0)  # Default to 3.0 if missing
            overall_score = validated_result.get(
                "overall_score", 3.0
            )  # Default to 3.0 if missing

            # Add debug information
            print(
                f"Extracted scores - Correctness: {correctness}, Completeness: {completeness}, Clarity: {clarity}"
            )

        except Exception as e:
            print(f"Error evaluating response: {str(e)}")
            correctness = completeness = clarity = overall_score = 0
            validated_result = {"scores": {}, "overall_score": 0, "reasoning": str(e)}
            latency_metrics = {
                "prompt_processing_ms": prompt_processing_ms,
                "total_evaluation_ms": prompt_processing_ms,
            }

        # Format actual response as string for storage
        formatted_actual = ""
        if actual_response_type == "command":
            command = self._get_attribute_safely(actual_response, "command", "")
            explanation = self._get_attribute_safely(actual_response, "explanation", "")
            security_notes = self._get_attribute_safely(
                actual_response, "security_notes", ""
            )

            formatted_actual = f"Command: {command}\n\nExplanation: {explanation}"
            if security_notes:
                formatted_actual += f"\n\nSecurity Notes: {security_notes}"
        else:
            answer = self._get_attribute_safely(actual_response, "answer", "")
            sources = self._get_attribute_safely(actual_response, "sources", [])
            sources_str = ", ".join(sources) if sources else "No sources provided"

            formatted_actual = f"Information: {answer}\n\nSources: {sources_str}"

        # Create result record
        result = {
            "query": question,
            "expected_response": expected_response,
            "actual_response": formatted_actual,  # Store formatted actual response
            "actual_response_raw": self._convert_to_serializable(
                actual_response
            ),  # Convert to serializable form
            "response_type": actual_response_type,
            "domain": domain,
            "type": query_type,
            "evaluation": validated_result,
            "correctness": correctness,
            "completeness": completeness,
            "clarity": clarity,
            "overall_score": overall_score,
            "latency_ms": latency_metrics,
            "timestamp": datetime.now().isoformat(),
        }

        # Print the evaluation details including explanations
        self._print_evaluation_details(result)

        return result

    def _get_attribute_safely(self, obj: Any, attr: str, default: Any = None) -> Any:
        """Safely get an attribute from an object, whether it's a dict or a model.

        Args:
            obj: The object to get the attribute from
            attr: The attribute name to get
            default: The default value to return if the attribute is not found

        Returns:
            The attribute value or the default
        """
        if obj is None:
            return default

        # If it's a dictionary, use get method
        if isinstance(obj, dict):
            return obj.get(attr, default)

        # If it's a model with attributes, use getattr
        if hasattr(obj, attr):
            return getattr(obj, attr)

        # If it's a model with model_dump method (Pydantic v2+)
        if hasattr(obj, "model_dump"):
            return obj.model_dump().get(attr, default)

        # If it's a model with dict method (Pydantic v1)
        if hasattr(obj, "dict"):
            return obj.dict().get(attr, default)

        # If all else fails, return the default
        return default

    def _print_evaluation_details(self, result: Dict[str, Any]) -> None:
        """Print detailed evaluation results for a sample.

        Args:
            result: The evaluation result dictionary
        """
        evaluation = result.get("evaluation", {})
        scores = evaluation.get("scores", {})

        correctness = scores.get("correctness", 0)
        completeness = scores.get("completeness", 0)
        clarity = scores.get("clarity", 0)
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

        print(f"\nClarity: {clarity}/5")
        print(
            f"Explanation: {evaluation.get('clarity_explanation', 'No explanation provided')}"
        )

        print(f"\nOverall Score: {overall:.1f}/5.0")
        print(f"Reasoning: {evaluation.get('reasoning', 'No reasoning provided')}")

        print(
            f"\nLatency: Processing={result.get('latency_ms', {}).get('prompt_processing_ms', 0):.1f}ms, "
            f"Evaluation={result.get('latency_ms', {}).get('total_evaluation_ms', 0):.1f}ms"
        )
        print("=" * 50)

    def run_evaluation(
        self,
        start_index: int = 0,
        end_index: Optional[int] = None,
        batch_size: int = 5,
        continue_from: Optional[str] = None,
        output_path: Optional[str] = None,
        verbose: bool = False,
    ) -> List[Dict]:
        """Run the evaluation on the dataset using the actual OS Assistant.

        Args:
            start_index: Index of the first sample to evaluate
            end_index: Index of the last sample to evaluate (None for all)
            batch_size: Number of samples to evaluate before saving results
            continue_from: Path to existing evaluation file to continue from
            output_path: Path to save the evaluation results
            verbose: Whether to print detailed progress information

        Returns:
            List of evaluation results
        """
        # Load dataset if not already loaded
        if not self.dataset:
            self.load_dataset()

        # Set output path
        if output_path:
            self.output_path = output_path
        else:
            # Generate default output path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_name = os.path.splitext(os.path.basename(self.dataset_path))[0]
            self.output_path = os.path.join(RESULTS_DIR, f"{dataset_name}_eval.json")

        # Continue from existing evaluation if requested
        if continue_from:
            try:
                with open(continue_from, "r", encoding="utf-8") as f:
                    existing_eval = json.load(f)
                    self.results = existing_eval.get("results", [])
                    print(
                        f"Continuing evaluation with {len(self.results)} existing results"
                    )
            except Exception as e:
                print(f"Error loading existing evaluation: {str(e)}")
                self.results = []

        # Determine evaluation range
        samples = self.dataset.samples[start_index:end_index]
        total_samples = len(samples)

        # Skip samples that have already been evaluated
        evaluated_queries = {result.get("query") for result in self.results}
        samples_to_evaluate = [
            sample for sample in samples if sample.question not in evaluated_queries
        ]

        # Print evaluation statistics
        print(f"Total samples: {total_samples}")
        print(f"Already evaluated: {len(evaluated_queries)}")
        print(f"Samples to evaluate: {len(samples_to_evaluate)}")

        # Evaluate samples in batches
        for i, sample in enumerate(samples_to_evaluate):
            # Evaluate the sample
            print(
                f"Evaluating sample {i+1}/{len(samples_to_evaluate)}: {sample.question[:50]}..."
            )
            try:
                sample_dict = (
                    sample.model_dump() if hasattr(sample, "model_dump") else sample
                )
                result = self.evaluate_sample(sample_dict)
                self.results.append(result)

                # Print evaluation result if verbose
                if verbose:
                    scores = result.get("evaluation", {}).get("scores", {})
                    print(f"  Correctness: {scores.get('correctness', 0):.2f}")
                    print(f"  Completeness: {scores.get('completeness', 0):.2f}")
                    print(f"  Clarity: {scores.get('clarity', 0):.2f}")
                    print(f"  Overall: {result.get('overall_score', 0):.2f}")

                # Save batch results periodically
                if (i + 1) % batch_size == 0:
                    self._save_interim_results(i + 1, len(samples_to_evaluate))

            except Exception as e:
                print(f"Error evaluating sample {i+1}: {str(e)}")
                continue

        # Save final results
        self.save_results()
        return self.results

    def _save_interim_results(self, current_sample: int, total_samples: int):
        """Save interim results during evaluation.

        Args:
            current_sample: Current sample index
            total_samples: Total number of samples to evaluate
        """
        try:
            # Generate running metrics
            running_metrics = self._generate_running_metrics(
                current_sample, total_samples
            )

            # Create interim results dictionary
            interim_results = {
                "timestamp": datetime.now().isoformat(),
                "dataset": self.dataset_path,
                "progress": f"{current_sample}/{total_samples}",
                "running_metrics": (
                    running_metrics.model_dump()
                    if hasattr(running_metrics, "model_dump")
                    else running_metrics
                ),
                "results": self.results,
            }

            # Convert to JSON serializable format
            serializable_results = self._convert_to_serializable(interim_results)

            # Save interim results
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(serializable_results, f, indent=2)

            print(f"Saved interim results ({current_sample}/{total_samples})")
            print(
                f"Running metrics: Avg score: {running_metrics.average_score:.2f}, Correctness: {running_metrics.average_correctness:.2f}, Completeness: {running_metrics.average_completeness:.2f}"
            )

        except Exception as e:
            print(f"Error saving interim results: {str(e)}")

    def _generate_running_metrics(
        self, current_sample: int, total_samples: int
    ) -> RunningMetrics:
        """Generate running metrics for the evaluation.

        Args:
            current_sample: Current sample index
            total_samples: Total number of samples to evaluate

        Returns:
            Running metrics for the evaluation
        """
        if not self.results:
            return RunningMetrics(
                total_evaluated=0,
                total_samples=total_samples,
                average_score=0,
                average_correctness=0,
                average_completeness=0,
                average_clarity=0,
                timestamp=datetime.now().isoformat(),
            )

        # Calculate average scores
        correctness_scores = [r.get("correctness", 0) for r in self.results]
        completeness_scores = [r.get("completeness", 0) for r in self.results]
        clarity_scores = [r.get("clarity", 0) for r in self.results]
        overall_scores = [r.get("overall_score", 0) for r in self.results]

        # Calculate latency metrics
        latency_metrics = {}
        prompt_processing_times = []
        llm_evaluation_times = []
        total_evaluation_times = []

        for result in self.results:
            latency = result.get("latency_ms", {})
            if latency:
                prompt_processing_times.append(latency.get("prompt_processing_ms", 0))
                if "total_evaluation_ms" in latency:
                    total_evaluation_times.append(latency.get("total_evaluation_ms", 0))

                # Add all LLM evaluation times
                for key, value in latency.items():
                    if key.endswith("_evaluation_ms") and key != "total_evaluation_ms":
                        llm_evaluation_times.append(value)

        # Calculate average latencies
        if prompt_processing_times:
            latency_metrics["avg_prompt_processing_ms"] = sum(
                prompt_processing_times
            ) / len(prompt_processing_times)
        if llm_evaluation_times:
            latency_metrics["avg_llm_evaluation_ms"] = sum(llm_evaluation_times) / len(
                llm_evaluation_times
            )
        if total_evaluation_times:
            latency_metrics["avg_total_evaluation_ms"] = sum(
                total_evaluation_times
            ) / len(total_evaluation_times)

        # Create running metrics
        return RunningMetrics(
            total_evaluated=len(self.results),
            total_samples=total_samples,
            average_score=(
                sum(overall_scores) / len(overall_scores) if overall_scores else 0
            ),
            average_correctness=(
                sum(correctness_scores) / len(correctness_scores)
                if correctness_scores
                else 0
            ),
            average_completeness=(
                sum(completeness_scores) / len(completeness_scores)
                if completeness_scores
                else 0
            ),
            average_clarity=(
                sum(clarity_scores) / len(clarity_scores) if clarity_scores else 0
            ),
            timestamp=datetime.now().isoformat(),
            latency_metrics=latency_metrics,
        )

    def save_results(self):
        """Save the evaluation results to a file."""
        try:
            # Generate summary
            summary = self.generate_summary()

            # Create final results dictionary
            final_results = {
                "timestamp": datetime.now().isoformat(),
                "dataset": self.dataset_path,
                "summary": (
                    summary.model_dump() if hasattr(summary, "model_dump") else summary
                ),
                "results": self.results,
            }

            # Convert to JSON serializable format
            serializable_results = self._convert_to_serializable(final_results)

            # Save final results
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(serializable_results, f, indent=2)

            print(f"Saved final results to {self.output_path}")

        except Exception as e:
            print(f"Error saving final results: {str(e)}")

    def generate_summary(self) -> EvaluationSummary:
        """Generate a summary of the evaluation results.

        Returns:
            Summary of the evaluation results
        """
        if not self.results:
            return EvaluationSummary(
                total_samples=0,
                average_score=0,
                average_correctness=0,
                average_completeness=0,
                average_clarity=0,
                domain_scores={},
                latency_metrics={},
                detailed_results=[],
            )

        # Calculate average scores
        correctness_scores = [r.get("correctness", 0) for r in self.results]
        completeness_scores = [r.get("completeness", 0) for r in self.results]
        clarity_scores = [r.get("clarity", 0) for r in self.results]
        overall_scores = [r.get("overall_score", 0) for r in self.results]

        # Calculate scores by domain
        domains = {}
        for result in self.results:
            domain = result.get("domain", "unknown")
            if domain not in domains:
                domains[domain] = []
            domains[domain].append(result.get("overall_score", 0))

        domain_scores = {
            domain: sum(scores) / len(scores) if scores else 0
            for domain, scores in domains.items()
        }

        # Calculate scores by query type
        command_scores = [
            r.get("overall_score", 0)
            for r in self.results
            if r.get("type") == "command"
        ]
        information_scores = [
            r.get("overall_score", 0)
            for r in self.results
            if r.get("type") == "information"
        ]

        # Calculate latency metrics
        latency_metrics = {}
        prompt_processing_times = []
        llm_evaluation_times = []
        total_evaluation_times = []

        for result in self.results:
            latency = result.get("latency_ms", {})
            if latency:
                prompt_processing_times.append(latency.get("prompt_processing_ms", 0))
                if "total_evaluation_ms" in latency:
                    total_evaluation_times.append(latency.get("total_evaluation_ms", 0))

                # Add all LLM evaluation times
                for key, value in latency.items():
                    if key.endswith("_evaluation_ms") and key != "total_evaluation_ms":
                        llm_evaluation_times.append(value)

        # Calculate average latencies
        if prompt_processing_times:
            latency_metrics["avg_prompt_processing_ms"] = sum(
                prompt_processing_times
            ) / len(prompt_processing_times)
        if llm_evaluation_times:
            latency_metrics["avg_llm_evaluation_ms"] = sum(llm_evaluation_times) / len(
                llm_evaluation_times
            )
        if total_evaluation_times:
            latency_metrics["avg_total_evaluation_ms"] = sum(
                total_evaluation_times
            ) / len(total_evaluation_times)
            latency_metrics["min_total_ms"] = min(total_evaluation_times)
            latency_metrics["max_total_ms"] = max(total_evaluation_times)

        # Create summary
        summary = EvaluationSummary(
            total_samples=len(self.results),
            average_score=(
                sum(overall_scores) / len(overall_scores) if overall_scores else 0
            ),
            average_correctness=(
                sum(correctness_scores) / len(correctness_scores)
                if correctness_scores
                else 0
            ),
            average_completeness=(
                sum(completeness_scores) / len(completeness_scores)
                if completeness_scores
                else 0
            ),
            average_clarity=(
                sum(clarity_scores) / len(clarity_scores) if clarity_scores else 0
            ),
            command_average=(
                sum(command_scores) / len(command_scores) if command_scores else None
            ),
            information_average=(
                sum(information_scores) / len(information_scores)
                if information_scores
                else None
            ),
            domain_scores=domain_scores,
            latency_metrics=latency_metrics,
            detailed_results=self.results,
        )

        return summary
