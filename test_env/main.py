import os
import sys
import time
import argparse
import random

from file_operations import FileSystemOperations
from content_generator import ContentGenerator
from scheduler import OperationScheduler


def main():
    parser = argparse.ArgumentParser(
        description="OS Assistant File System Test Environment"
    )
    parser.add_argument(
        "--base-path",
        default=None,
        help="Base path for test data (defaults to the test_env/data folder relative to this script)",
    )
    parser.add_argument(
        "--min-interval",
        type=int,
        default=5,
        help="Minimum interval between operations (seconds)",
    )
    parser.add_argument(
        "--max-interval",
        type=int,
        default=30,
        help="Maximum interval between operations (seconds)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="Duration to run (minutes, 0 for indefinite)",
    )

    args = parser.parse_args()

    # Setup components
    if not args.base_path:
        # Default to the data folder inside this test_env directory
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
    else:
        base_path = os.path.abspath(args.base_path)

    print(f"Initializing test environment in: {base_path}")
    content_gen = ContentGenerator()
    file_ops = FileSystemOperations(base_path)
    scheduler = OperationScheduler(
        file_ops,
        content_gen,
        min_interval=args.min_interval,
        max_interval=args.max_interval,
    )

    # Create initial structure with more natural folder names
    print("Creating initial folder structure...")
    initial_folders = ["documents", "projects", "config", "temp"]

    # Add some random initial folders for variety
    adjectives = ["main", "shared", "personal", "important", "archived"]
    nouns = ["data", "files", "resources", "assets", "backups"]

    for _ in range(3):  # Add 3 random folders
        folder_name = f"{random.choice(adjectives)}_{random.choice(nouns)}"
        initial_folders.append(folder_name)

    for folder in initial_folders:
        file_ops.create_folder(folder)
        scheduler.folders_created.add(folder)

    # Create some initial files with more meaningful names
    print("Creating initial files...")
    initial_files = [
        ("documents/readme.md", content_gen.generate_markdown_file()),
        ("projects/sample.py", content_gen.generate_python_file()),
        ("config/settings.json", content_gen.generate_json_file()),
        ("notes.txt", content_gen.generate_text_file()),
        (
            f"{random.choice(initial_folders)}/important_data.csv",
            content_gen.generate_csv_file(),
        ),
        (
            f"{random.choice(initial_folders)}/system.log",
            content_gen.generate_log_file(),
        ),
    ]

    for file_path, content in initial_files:
        file_ops.create_file(file_path, content)

    # Start the scheduler
    print(
        f"Starting scheduler (min: {args.min_interval}s, max: {args.max_interval}s)..."
    )
    scheduler.start()

    try:
        if args.duration > 0:
            duration_seconds = args.duration * 60
            print(f"Running for {args.duration} minutes...")
            time.sleep(duration_seconds)
            scheduler.stop()
            print("Test completed.")
        else:
            print("Running indefinitely. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping scheduler...")
        scheduler.stop()
        print(
            f"Test environment stopped. Total operations: {scheduler.operation_count}"
        )


if __name__ == "__main__":
    main()
