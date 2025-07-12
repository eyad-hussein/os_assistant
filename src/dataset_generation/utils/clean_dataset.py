import json
import os
import shutil
import sys
from datetime import datetime


def clean_dataset(dataset_path: str, create_backup: bool = True) -> None:
    """
    Remove entries from the dataset where execution_result is "No output"
    and generated_type is "code_execution"

    Args:
        dataset_path: Path to the dataset file
        create_backup: Whether to create a backup of the original file
    """
    # Check if file exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found at {dataset_path}")
        return

    try:
        # Create backup if requested
        if create_backup:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{dataset_path}.{timestamp}.bak"
            shutil.copy2(dataset_path, backup_path)
            print(f"Created backup at: {backup_path}")

        # Read the dataset
        with open(dataset_path, encoding="utf-8") as f:
            dataset = json.load(f)

        # Get initial structure info
        print(f"Original dataset type: {type(dataset).__name__}")

        # Handle the specific structure with metadata and samples
        if isinstance(dataset, dict) and "metadata" in dataset and "samples" in dataset:
            print("Found standard dataset structure with metadata and samples")

            # Get the samples list
            samples = dataset["samples"]
            metadata = dataset["metadata"]

            if not isinstance(samples, list):
                print(f"Error: 'samples' is not a list but a {type(samples).__name__}")
                return

            # Get initial count
            initial_count = len(samples)
            print(f"Original dataset has {initial_count} samples")

            # Filter out problematic entries
            filtered_samples = []
            removed_samples = []

            for sample in samples:
                if not isinstance(sample, dict):
                    print(
                        f"Warning: Found non-dictionary sample: {type(sample).__name__}. Skipping."
                    )
                    continue

                if (
                    sample.get("execution_result") == "No output"
                    and sample.get("generated_type") == "code_execution"
                ):
                    removed_samples.append(sample.get("question", "Unknown question"))
                else:
                    filtered_samples.append(sample)

            # Get final count
            final_count = len(filtered_samples)
            removed_count = initial_count - final_count

            # Update the samples list
            dataset["samples"] = filtered_samples

            # Update metadata
            if "num_samples" in metadata:
                metadata["num_samples"] = final_count

            # Update generation_type_distribution if it exists
            if (
                "generation_type_distribution" in metadata
                and "code_execution" in metadata["generation_type_distribution"]
            ):
                code_exec_count = metadata["generation_type_distribution"][
                    "code_execution"
                ]
                metadata["generation_type_distribution"]["code_execution"] = max(
                    0, code_exec_count - removed_count
                )

            # Update updated_at field
            metadata["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        else:
            # Handle other formats
            print(
                "Warning: Dataset does not have the expected structure with metadata and samples"
            )

            # For dictionary format, extract entries
            is_dict_format = isinstance(dataset, dict)
            is_list_format = isinstance(dataset, list)

            if not (is_dict_format or is_list_format):
                print(
                    f"Error: Dataset is neither a dictionary nor a list: {type(dataset).__name__}"
                )
                return

            if is_dict_format:
                # Try to find entries - check common keys for stored data
                entries = None
                possible_keys = ["data", "items", "questions", "entries", "samples"]

                for key in possible_keys:
                    if key in dataset and isinstance(dataset[key], list):
                        entries = dataset[key]
                        print(f"Found entries in '{key}' key")
                        break

                # If no entries found in common keys, check if values are entries
                if entries is None:
                    # Check if the dictionary values are the entries
                    first_value = next(iter(dataset.values()), None)
                    if isinstance(first_value, dict) and "question" in first_value:
                        print("Dataset appears to be a dictionary of entries")
                        entries = list(dataset.values())
                    else:
                        # Couldn't find entries in the dictionary
                        print("Error: Could not locate entries in the dictionary")
                        print("Dictionary keys:", list(dataset.keys())[:10])
                        print(
                            "First value type:",
                            type(first_value).__name__ if first_value else "None",
                        )
                        return
            else:
                # List format - entries are the list itself
                entries = dataset

            # Get initial count
            initial_count = len(entries)
            print(f"Original dataset has {initial_count} entries")

            # Filter out problematic entries
            filtered_entries = []
            removed_entries = []

            for entry in entries:
                # Check if entry is a dictionary
                if not isinstance(entry, dict):
                    print(
                        f"Warning: Found non-dictionary entry: {type(entry).__name__}. Skipping."
                    )
                    continue

                if (
                    entry.get("execution_result") == "No output"
                    and entry.get("generated_type") == "code_execution"
                ):
                    removed_entries.append(entry.get("question", "Unknown question"))
                else:
                    filtered_entries.append(entry)

            # Get final count
            final_count = len(filtered_entries)
            removed_count = initial_count - final_count

            # Update the dataset with filtered entries
            if is_dict_format:
                # If entries were found in a key, update that key
                for key in possible_keys:
                    if key in dataset and isinstance(dataset[key], list):
                        dataset[key] = filtered_entries
                        break
                # If entries were the values, rebuild the dictionary
                if not any(key in dataset for key in possible_keys):
                    # This is more complex and would require preserving keys
                    # For simplicity, we'll convert to a list format
                    print("Converting dictionary of entries to list format")
                    dataset = filtered_entries
            else:
                # List format - replace the entire list
                dataset = filtered_entries

        # Write filtered dataset back to file
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        print("\nDataset cleaning complete!")
        print(f"Removed {removed_count} entries with 'No output' execution results")
        print(f"New dataset size: {final_count} entries")
        print(f"Final dataset type: {type(dataset).__name__}")

        # Print removed questions if any
        if removed_samples if "removed_samples" in locals() else removed_entries:
            removed_questions = (
                removed_samples if "removed_samples" in locals() else removed_entries
            )
            print("\nRemoved questions:")
            for i, question in enumerate(removed_questions[:10], 1):
                print(f"{i}. {question}")

            if len(removed_questions) > 10:
                print(f"...and {len(removed_questions) - 10} more")

    except json.JSONDecodeError:
        print("Error: Invalid JSON in the dataset file")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Default path
    default_path = os.path.join("datasets", "file_system_dataset.json")

    # Use command line argument if provided, otherwise use default
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else default_path

    # Run the cleaning process
    clean_dataset(dataset_path)
