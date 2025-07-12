import json
import os
import time
from typing import Any

# Import our components
from dataset_generation.config.config import (
    DATASET_OUTPUT_DIR,
    DEFAULT_DATASET_FILENAME,
)
from dataset_generation.core.log_sampler import SequentialLogSampler
from dataset_generation.core.question_generator import QuestionGenerator
from dataset_generation.core.similarity_checker import (
    QuestionSimilarityChecker,
)


def generate_log_based_questions(
    generator: QuestionGenerator,
    sampler: SequentialLogSampler,
    similarity_checker: QuestionSimilarityChecker,
    num_samples: int = 10,
    previous_questions: list[str] = None,
) -> list[dict[str, Any]]:
    """
    Generate questions based on file system logs.

    Args:
        generator: QuestionGenerator instance
        sampler: SequentialLogSampler instance
        similarity_checker: QuestionSimilarityChecker instance
        num_samples: Number of log-based questions to generate
        previous_questions: Optional list of recent questions to avoid duplicating

    Returns:
        List of log-based questions
    """
    all_questions = []

    # Generate questions until we have enough
    while len(all_questions) < num_samples:
        # Get sequential logs for question generation
        logs = sampler.get_sequential_logs(count=3)
        if not logs:
            print("[WARNING] Could not retrieve logs for question generation")
            break

        # Generate questions from these logs, passing recent questions for context
        questions = generator.generate_questions_from_logs(
            logs, previous_questions=previous_questions
        )

        # Filter out duplicate questions
        unique_questions = similarity_checker.filter_duplicates(questions)

        # Add unique questions to our result
        all_questions.extend(unique_questions)

        # Stop if we have enough questions
        if len(all_questions) >= num_samples:
            break

    # Return only the requested number of questions
    return all_questions[:num_samples]


def generate_random_questions(
    generator: QuestionGenerator,
    similarity_checker: QuestionSimilarityChecker,
    num_samples: int = 10,
) -> list[dict[str, Any]]:
    """
    Generate random file system questions not tied to specific logs.

    Args:
        generator: QuestionGenerator instance
        similarity_checker: QuestionSimilarityChecker instance
        num_samples: Number of random questions to generate

    Returns:
        List of random questions
    """
    # Generate more questions than needed to account for filtering
    questions = generator.generate_random_questions(num_questions=num_samples * 2)

    # Filter out duplicate questions
    unique_questions = similarity_checker.filter_duplicates(questions)

    # Return only the requested number of questions
    return unique_questions[:num_samples]


def generate_code_execution_questions(
    generator: QuestionGenerator,
    similarity_checker: QuestionSimilarityChecker,
    num_samples: int = 5,
) -> list[dict[str, Any]]:
    """
    Generate questions that require code execution to answer.

    Args:
        generator: QuestionGenerator instance
        similarity_checker: QuestionSimilarityChecker instance
        num_samples: Number of code execution questions to generate

    Returns:
        List of code execution questions
    """
    # Generate more questions than needed to account for filtering
    questions = generator.generate_code_execution_questions(
        num_questions=num_samples * 2
    )

    # Filter out duplicate questions
    unique_questions = similarity_checker.filter_duplicates(questions)

    # Return only the requested number of questions
    return unique_questions[:num_samples]


def save_dataset(
    questions: list[dict[str, Any]],
    filename: str = DEFAULT_DATASET_FILENAME,
    additional_metadata: dict[str, Any] | None = None,
    append: bool = True,
) -> None:
    """
    Save the dataset to a JSON file, appending to existing file if it exists.

    Args:
        questions: List of question dictionaries
        filename: Name of the output file
        additional_metadata: Additional metadata to include
        append: Whether to append to existing file or overwrite
    """
    # Ensure the output directory exists
    os.makedirs(DATASET_OUTPUT_DIR, exist_ok=True)

    # Full path to the output file
    output_path = os.path.join(DATASET_OUTPUT_DIR, filename)

    # Calculate the generation type distribution
    generation_type_distribution = {}
    for question in questions:
        gen_type = question.get("generated_type", "unknown")
        if gen_type in generation_type_distribution:
            generation_type_distribution[gen_type] += 1
        else:
            generation_type_distribution[gen_type] = 1

    # Calculate the question type distribution
    question_type_distribution = {}
    for question in questions:
        q_type = question.get("type", "unknown")
        if q_type in question_type_distribution:
            question_type_distribution[q_type] += 1
        else:
            question_type_distribution[q_type] = 1

    # Initialize with default structure
    dataset = {
        "metadata": {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_samples": len(questions),
            "description": "File System Command Dataset for OS Assistant",
            "domain": "file_system",
            "question_type_distribution": question_type_distribution,
            "generation_type_distribution": generation_type_distribution,
        },
        "samples": questions,
    }

    # Add additional metadata if provided
    if additional_metadata:
        dataset["metadata"].update(additional_metadata)

    # Check if we should append to existing file
    if append and os.path.exists(output_path):
        try:
            # Load existing dataset
            with open(output_path) as f:
                existing_dataset = json.load(f)

            # Update metadata
            existing_dataset["metadata"]["updated_at"] = time.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            existing_dataset["metadata"]["num_samples"] += len(questions)

            # Update generation type distribution
            existing_distribution = existing_dataset["metadata"].get(
                "generation_type_distribution", {}
            )
            for gen_type, count in generation_type_distribution.items():
                if gen_type in existing_distribution:
                    existing_distribution[gen_type] += count
                else:
                    existing_distribution[gen_type] = count
            existing_dataset["metadata"]["generation_type_distribution"] = (
                existing_distribution
            )

            # Update question type distribution
            existing_q_distribution = existing_dataset["metadata"].get(
                "question_type_distribution", {}
            )
            for q_type, count in question_type_distribution.items():
                if q_type in existing_q_distribution:
                    existing_q_distribution[q_type] += count
                else:
                    existing_q_distribution[q_type] = count
            existing_dataset["metadata"]["question_type_distribution"] = (
                existing_q_distribution
            )

            # Add the new questions to existing samples
            existing_dataset["samples"].extend(questions)

            # Update with additional metadata if provided
            if additional_metadata:
                existing_dataset["metadata"].update(additional_metadata)

            # Use the updated dataset
            dataset = existing_dataset
            print(
                f"Appended {len(questions)} questions to existing dataset at {output_path}"
            )
        except Exception as e:
            print(
                f"[WARNING] Error appending to existing dataset: {str(e)}. Creating new file."
            )

    # Save the dataset
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Saved dataset with {len(questions)} questions to {output_path}")
    print(
        f"Generation type distribution: {dataset['metadata']['generation_type_distribution']}"
    )
    print(
        f"Question type distribution: {dataset['metadata']['question_type_distribution']}"
    )
