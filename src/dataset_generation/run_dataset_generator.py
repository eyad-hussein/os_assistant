import argparse
import json
import os
import time

from tracer.config import LogDomain

from dataset_generation.config.config import (
    DATASET_OUTPUT_DIR,
    SIMILARITY_THRESHOLD,
)
from dataset_generation.core.generate_dataset import (
    generate_code_execution_questions,
    generate_log_based_questions,
    generate_random_questions,
    save_dataset,
)
from dataset_generation.core.log_sampler import SequentialLogSampler
from dataset_generation.core.question_generator import QuestionGenerator
from dataset_generation.core.similarity_checker import (
    QuestionSimilarityChecker,
    check_duplicate_with_dataset,
)


def parse_args():
    """Parse command line arguments for dataset generation"""
    parser = argparse.ArgumentParser(
        description="Generate a dataset of file system questions"
    )

    parser.add_argument(
        "--log-questions",
        type=int,
        default=1,
        help="Number of log-based questions to generate (default: 1)",
    )

    parser.add_argument(
        "--random-questions",
        type=int,
        default=1,
        help="Number of random questions to generate (default: 1)",
    )

    parser.add_argument(
        "--code-questions",
        type=int,
        default=0,
        help="Number of code execution questions to generate (default: 0)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="file_system_dataset.json",
        help="Output filename (default: file_system_dataset.json)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of questions to process in each batch (default: 5)",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing dataset instead of appending",
    )

    return parser.parse_args()


def process_question_batch(questions, filename, additional_metadata, append=True):
    """Process and save a batch of questions"""
    # Check for duplicates with existing dataset
    output_path = os.path.join(DATASET_OUTPUT_DIR, filename)
    if append and os.path.exists(output_path):
        questions = check_duplicate_with_dataset(questions, output_path)

    # Save the batch
    save_dataset(
        questions=questions,
        filename=filename,
        additional_metadata=additional_metadata,
        append=append,
    )
    return len(questions)


def get_recent_questions(filename, count=5):
    """Get the most recent questions from an existing dataset"""
    output_path = os.path.join(DATASET_OUTPUT_DIR, filename)
    recent_questions = []

    if os.path.exists(output_path):
        try:
            with open(output_path, "r") as f:
                dataset = json.load(f)
                samples = dataset.get("samples", [])

                # Get the questions from the most recent samples
                for sample in samples[-count:]:
                    if "question" in sample:
                        recent_questions.append(sample["question"])
        except Exception as e:
            print(f"[WARNING] Error getting recent questions: {str(e)}")

    return recent_questions


def main():
    """Generate a comprehensive dataset of file system questions"""
    # Parse command line arguments
    args = parse_args()

    start_time = time.time()
    print("=== Starting dataset generation ===")
    print(
        f"Parameters: {args.log_questions} log questions, {args.random_questions} random questions, "
        f"{args.code_questions} code questions, batch size: {args.batch_size}"
    )
    print(
        "All questions will focus on the D:\\Graduation_Project_Test_Environment directory"
    )

    # Generate a timestamped filename for this run
    filename = args.output

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
            if f.startswith("file_system_dataset") and f.endswith(".json")
        ]
        if dataset_files:
            dataset_files.sort(reverse=True)  # Latest first
            existing_dataset_path = os.path.join(DATASET_OUTPUT_DIR, dataset_files[0])
            print(f"Found existing dataset: {existing_dataset_path}")
    except Exception as e:
        print(f"[WARNING] Error looking for existing datasets: {str(e)}")

    # Add timing information to metadata
    additional_metadata = {
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "generation_parameters": {
            "log_questions": args.log_questions,
            "random_questions": args.random_questions,
            "code_questions": args.code_questions,
        },
        "focus_path": "D:\\Graduation_Project_Test_Environment",  # Add focus path to metadata
    }

    # First batch should overwrite if --overwrite is specified
    first_batch = True

    # Track the total number of questions generated by type
    total_log_questions = 0
    total_random_questions = 0
    total_code_questions = 0

    # Keep track of all generated questions to avoid duplication
    all_generated_questions = get_recent_questions(filename)
    print(
        f"Retrieved {len(all_generated_questions)} recent questions from existing dataset"
    )

    # 1. Generate log-based questions in batches
    remaining_log_questions = args.log_questions
    while remaining_log_questions > 0:
        batch_size = min(remaining_log_questions, args.batch_size)
        print(f"\nGenerating batch of {batch_size} log-based questions...")

        log_questions = generate_log_based_questions(
            generator=generator,
            sampler=sampler,
            similarity_checker=similarity_checker,
            num_samples=batch_size,
            previous_questions=all_generated_questions,  # Pass recent questions
        )

        # Save this batch
        saved_count = process_question_batch(
            log_questions,
            filename,
            additional_metadata,
            append=not (first_batch and args.overwrite),
        )

        # Update our list of generated questions
        all_generated_questions.extend([q["question"] for q in log_questions])

        total_log_questions += saved_count
        remaining_log_questions -= batch_size
        first_batch = False

    # 2. Generate random questions in batches
    remaining_random_questions = args.random_questions
    while remaining_random_questions > 0:
        batch_size = min(remaining_random_questions, args.batch_size)
        print(f"\nGenerating batch of {batch_size} random questions...")

        random_questions = generate_random_questions(
            generator=generator,
            similarity_checker=similarity_checker,
            num_samples=batch_size,
        )

        # Save this batch
        saved_count = process_question_batch(
            random_questions,
            filename,
            additional_metadata,
            append=not (first_batch and args.overwrite),
        )

        total_random_questions += saved_count
        remaining_random_questions -= batch_size
        first_batch = False

    # 3. Generate code execution questions in batches
    remaining_code_questions = args.code_questions
    while remaining_code_questions > 0:
        batch_size = min(remaining_code_questions, args.batch_size)
        print(f"\nGenerating batch of {batch_size} code execution questions...")

        code_questions = generate_code_execution_questions(
            generator=generator,
            similarity_checker=similarity_checker,
            num_samples=batch_size,
        )

        # Save this batch
        saved_count = process_question_batch(
            code_questions,
            filename,
            additional_metadata,
            append=not (first_batch and args.overwrite),
        )

        total_code_questions += saved_count
        remaining_code_questions -= batch_size
        first_batch = False

    elapsed_time = time.time() - start_time

    print("\n=== Dataset generation completed ===")
    print(f"Generated and saved questions in {elapsed_time/60:.1f} minutes:")
    print(f"- {total_log_questions} from logs")
    print(f"- {total_random_questions} random questions")
    print(f"- {total_code_questions} code execution questions")
    print(
        f"Total: {total_log_questions + total_random_questions + total_code_questions} questions"
    )


if __name__ == "__main__":
    main()
