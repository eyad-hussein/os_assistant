import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List

from os_assistant.os_assistant import OSAssistant

from ..utils.models import EvaluationDataset, EvaluationSummary
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

        # Process the question
        start_time = time.time()
        assistant.process_prompt(question)
        end_time = time.time()
        processing_time = end_time - start_time

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
                        "relevance": 0,
                        "clarity": 0,
                    },
                    "overall_score": 0.0,
                    "reasoning": "Assistant did not generate a final result",
                },
                "processing_time": processing_time,
            }

        # Get the actual response
        actual_response = final_result.get("response", {})
        actual_response_type = final_result.get("response_type", query_type)

        # Evaluate using LLM Judge
        evaluation = self.judge.evaluate(
            question=question,
            expected_response=expected_response,
            actual_response=actual_response,
            query_type=query_type,
        )

        # Create result record
        result = {
            "sample": sample,
            "actual_response": actual_response,
            "actual_response_type": actual_response_type,
            "evaluation": evaluation,
            "processing_time": processing_time,
        }

        return result

    def run_evaluation(self, num_samples: int | None = None) -> List[Dict[str, Any]]:
        """Run evaluation on the dataset.

        Args:
            num_samples: Number of samples to evaluate (None for all)

        Returns:
            List of evaluation results
        """
        if not self.dataset:
            self.load_dataset()

        samples = self.dataset.samples
        if num_samples:
            samples = samples[:num_samples]

        self.results = []
        for i, sample in enumerate(samples):
            print(f"\nEvaluating sample {i+1}/{len(samples)}")
            sample_dict = sample.model_dump()
            result = self.evaluate_sample(sample_dict)
            self.results.append(result)

        return self.results

    def generate_summary(self) -> EvaluationSummary:
        """Generate a summary of the evaluation results.

        Returns:
            Summary statistics
        """
        if not self.results:
            raise ValueError("No evaluation results to summarize")

        total_samples = len(self.results)
        all_scores = [
            r.get("evaluation", {}).get("overall_score", 0) for r in self.results
        ]
        avg_score = sum(all_scores) / total_samples if total_samples > 0 else 0

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
                    "overall_score": result.get("evaluation", {}).get(
                        "overall_score", 0
                    ),
                    "reasoning": result.get("evaluation", {}).get("reasoning", ""),
                }
            )

        # Create the summary
        summary = EvaluationSummary(
            total_samples=total_samples,
            average_score=avg_score,
            command_average=command_avg,
            information_average=info_avg,
            domain_scores=domain_avgs,
            detailed_results=detailed,
        )

        return summary

    def save_results(self, output_path: str) -> None:
        """Save the evaluation results to a file.

        Args:
            output_path: Path to save the results
        """
        if not self.results:
            raise ValueError("No evaluation results to save")

        # Generate summary
        summary = self.generate_summary()

        # Create full report
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": summary.model_dump(),
            "detailed_results": self.results,
        }

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save to file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(f"Evaluation results saved to {output_path}")
