import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List

from tracer.config import LogDomain

# Import our components
from os_assistant.dataset_generation.config import (
    DATASET_OUTPUT_DIR,
    DEFAULT_DATASET_FILENAME,
    NUM_SAMPLES_PER_DOMAIN,
    SIMILARITY_THRESHOLD,
)
from os_assistant.dataset_generation.core.log_sampler import SequentialLogSampler
from os_assistant.dataset_generation.core.question_generator import QuestionGenerator
from os_assistant.dataset_generation.core.similarity_checker import (
    QuestionSimilarityChecker,
    check_duplicate_with_dataset,
)


def generate_log_based_questions(
    generator: QuestionGenerator,
    sampler: SequentialLogSampler,
    similarity_checker: QuestionSimilarityChecker,
    num_samples: int = 20,
) -> List[Dict[str, Any]]:
    """Generate questions based on logs"""
    print(f"\n=== Generating {num_samples} log-based questions ===")
    questions = []
    attempts = 0
    max_attempts = num_samples * 3  # Allow for more failures due to duplicate filtering
    seen_questions = []  # Track questions for duplicate checking

    while len(questions) < num_samples and attempts < max_attempts:
        try:
            # Get a sequence of logs
            logs = sampler.get_sequential_logs(count=3)
            if not logs:
                print("[WARNING] Could not retrieve logs, skipping batch")
                attempts += 1
                continue

            # Generate questions from these logs
            batch_size = min(3, num_samples - len(questions))
            log_questions = generator.generate_questions_from_logs(
                logs=logs, num_questions=batch_size, domain_hint="file_system"
            )

            if not log_questions:
                print("[WARNING] No questions generated from logs")
                attempts += 1
                continue

            # Filter out duplicates
            filtered_batch = []
            for q in log_questions:
                question_text = q.get("question", "")
                if not question_text:
                    continue

                is_dup, similarity, similar_q = similarity_checker.is_duplicate(
                    question_text, seen_questions
                )

                if not is_dup:
                    filtered_batch.append(q)
                    seen_questions.append(question_text)

            # Enhance answers with RAG if any questions survived filtering
            if filtered_batch:
                enhanced_batch = generator.enhance_with_rag(filtered_batch)
                questions.extend(enhanced_batch)
                print(
                    f"Added {len(enhanced_batch)} unique questions after duplicate filtering"
                )

        except Exception as e:
            print(f"[ERROR] Error generating log-based questions: {str(e)}")

        attempts += 1
        time.sleep(1)  # Small pause to avoid overwhelming the system

    print(
        f"Generated {len(questions)}/{num_samples} log-based questions after {attempts} attempts"
    )
    return questions


def generate_random_questions(
    generator: QuestionGenerator,
    similarity_checker: QuestionSimilarityChecker,
    num_samples: int = 15,
) -> List[Dict[str, Any]]:
    """Generate random questions not based on logs"""
    print(f"\n=== Generating {num_samples} random questions ===")
    questions = []
    attempts = 0
    max_attempts = num_samples * 3
    seen_questions = []

    while len(questions) < num_samples and attempts < max_attempts:
        try:
            # Generate questions in small batches
            batch_size = min(5, num_samples - len(questions))
            random_questions = generator.generate_random_questions(
                num_questions=batch_size
            )

            if not random_questions:
                print("[WARNING] No random questions generated")
                attempts += 1
                continue

            # Filter duplicates
            filtered_batch = []
            for q in random_questions:
                question_text = q.get("question", "")
                if not question_text:
                    continue

                is_dup, similarity, similar_q = similarity_checker.is_duplicate(
                    question_text, seen_questions
                )

                if not is_dup:
                    filtered_batch.append(q)
                    seen_questions.append(question_text)

            questions.extend(filtered_batch)
            print(f"Added {len(filtered_batch)} unique random questions")

        except Exception as e:
            print(f"[ERROR] Error generating random questions: {str(e)}")

        attempts += 1
        time.sleep(1)

    print(
        f"Generated {len(questions)}/{num_samples} random questions after {attempts} attempts"
    )
    return questions


def generate_code_execution_questions(
    generator: QuestionGenerator,
    similarity_checker: QuestionSimilarityChecker,
    num_samples: int = 10,
) -> List[Dict[str, Any]]:
    """Generate questions that require code execution"""
    print(f"\n=== Generating {num_samples} code execution questions ===")
    questions = []
    attempts = 0
    max_attempts = num_samples * 3  # Code execution might fail more often
    seen_questions = []

    while len(questions) < num_samples and attempts < max_attempts:
        try:
            # Generate one question at a time for code execution
            batch_size = min(2, num_samples - len(questions))
            code_questions = generator.generate_code_execution_questions(
                num_questions=batch_size
            )

            if not code_questions:
                print("[WARNING] No code execution questions generated")
                attempts += 1
                continue

            # Filter duplicates
            filtered_batch = []
            for q in code_questions:
                question_text = q.get("question", "")
                if not question_text:
                    continue

                is_dup, similarity, similar_q = similarity_checker.is_duplicate(
                    question_text, seen_questions
                )

                if not is_dup:
                    filtered_batch.append(q)
                    seen_questions.append(question_text)

            questions.extend(filtered_batch)
            print(f"Added {len(filtered_batch)} unique code execution questions")

        except Exception as e:
            print(f"[ERROR] Error generating code execution questions: {str(e)}")

        attempts += 1
        time.sleep(2)  # Longer pause for code execution

    print(
        f"Generated {len(questions)}/{num_samples} code execution questions after {attempts} attempts"
    )
    return questions


def save_dataset(
    questions: List[Dict[str, Any]],
    filename: str = DEFAULT_DATASET_FILENAME,
    additional_metadata: Dict[str, Any] = None,
) -> str:
    """Save dataset to a JSON file and return the path"""
    print(f"\n=== Saving dataset to {filename} ===")

    # Create output directory if it doesn't exist
    os.makedirs(DATASET_OUTPUT_DIR, exist_ok=True)

    # Count question types
    question_types = {}
    generation_types = {}

    for q in questions:
        q_type = q.get("type", "unknown")
        question_types[q_type] = question_types.get(q_type, 0) + 1

        gen_type = q.get("generated_type", "log_based")
        generation_types[gen_type] = generation_types.get(gen_type, 0) + 1

    # Create dataset with detailed metadata
    metadata = {
        "created_at": datetime.now().isoformat(),
        "num_samples": len(questions),
        "description": "File System Command Dataset for OS Assistant",
        "domain": "file_system",
        "question_type_distribution": question_types,
        "generation_type_distribution": generation_types,
    }

    # Add any additional metadata
    if additional_metadata:
        metadata.update(additional_metadata)

    dataset = {
        "metadata": metadata,
        "samples": questions,
    }

    # Write to file
    output_path = os.path.join(DATASET_OUTPUT_DIR, filename)
    try:
        with open(output_path, "w") as f:
            json.dump(dataset, f, indent=2)
        print(
            f"Successfully saved dataset with {len(questions)} questions to {output_path}"
        )
        return output_path
    except Exception as e:
        print(f"[ERROR] Error saving dataset: {str(e)}")
        return ""


def main():
    """Generate a comprehensive dataset of file system questions"""
    start_time = time.time()
    print("=== Starting dataset generation ===")

    # Generate a timestamped filename for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"file_system_dataset_{timestamp}.json"
    output_path = os.path.join(DATASET_OUTPUT_DIR, filename)

    # Initialize components
    sampler = SequentialLogSampler(domain=LogDomain.FS)
    generator = QuestionGenerator()
    similarity_checker = QuestionSimilarityChecker(threshold=SIMILARITY_THRESHOLD)

    # Check for existing dataset to prevent duplicates
    existing_dataset_path = None
    try:
        # Look for most recent dataset file
        dataset_files = [
            f
            for f in os.listdir(DATASET_OUTPUT_DIR)
            if f.startswith("file_system_dataset_") and f.endswith(".json")
        ]
        if dataset_files:
            dataset_files.sort(reverse=True)  # Latest first
            existing_dataset_path = os.path.join(DATASET_OUTPUT_DIR, dataset_files[0])
            print(f"Found existing dataset: {existing_dataset_path}")
    except Exception as e:
        print(f"[WARNING] Error looking for existing datasets: {str(e)}")

    all_questions = []

    # 1. Generate log-based questions
    log_questions = generate_log_based_questions(
        generator=generator,
        sampler=sampler,
        similarity_checker=similarity_checker,
        num_samples=1,
    )
    all_questions.extend(log_questions)

    # 2. Generate random questions
    random_questions = generate_random_questions(
        generator=generator, similarity_checker=similarity_checker, num_samples=1
    )
    all_questions.extend(random_questions)

    # 3. Generate code execution questions
    code_questions = generate_code_execution_questions(
        generator=generator, similarity_checker=similarity_checker, num_samples=0
    )
    all_questions.extend(code_questions)

    # 4. Final check against existing dataset if available
    if existing_dataset_path:
        print(
            f"Checking for duplicates against existing dataset: {existing_dataset_path}"
        )
        all_questions = check_duplicate_with_dataset(
            all_questions, existing_dataset_path
        )

    # Save the complete dataset
    elapsed_time = time.time() - start_time

    # Add timing information to metadata
    additional_metadata = {
        "generation_time_seconds": elapsed_time,
        "generation_time_formatted": f"{elapsed_time/60:.1f} minutes",
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "filtered_duplicates": True,
        "rag_enhanced": True,
    }

    save_dataset(
        questions=all_questions,
        filename=filename,
        additional_metadata=additional_metadata,
    )

    print("\n=== Dataset generation completed ===")
    print(
        f"Generated {len(all_questions)} total questions in {elapsed_time/60:.1f} minutes"
    )
    print(f"- {len(log_questions)} from logs")
    print(f"- {len(random_questions)} random questions")
    print(f"- {len(code_questions)} code execution questions")


if __name__ == "__main__":
    main()
