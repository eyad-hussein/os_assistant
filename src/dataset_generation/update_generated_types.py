import json
import os


def update_dataset_generated_types(filepath):
    """
    Updates the dataset by setting 'log_based' as the generated_type
    for any entries missing this field and updates the metadata counts.
    """
    print(f"Loading dataset from: {filepath}")

    # Load the JSON file
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Count existing generated_type distributions
    generation_type_counts = {}
    # Also count question types
    question_type_counts = {}

    for sample in data["samples"]:
        # Count generation types
        gen_type = sample.get("generated_type")
        if gen_type:
            generation_type_counts[gen_type] = (
                generation_type_counts.get(gen_type, 0) + 1
            )

        # Count question types
        question_type = sample.get("type")
        if question_type:
            question_type_counts[question_type] = (
                question_type_counts.get(question_type, 0) + 1
            )

    print(f"Initial generation type distribution: {generation_type_counts}")
    print(f"Current question type distribution: {question_type_counts}")

    # Count entries that need to be updated
    entries_to_update = sum(
        1 for sample in data["samples"] if "generated_type" not in sample
    )
    print(f"Found {entries_to_update} entries without a generated_type")

    # Update entries without a generated_type
    for sample in data["samples"]:
        if "generated_type" not in sample:
            sample["generated_type"] = "log_based"

    # Count updated generation_type distributions
    updated_generation_type_counts = {}
    updated_question_type_counts = {}

    for sample in data["samples"]:
        # Count updated generation types
        gen_type = sample.get("generated_type")
        if gen_type:
            updated_generation_type_counts[gen_type] = (
                updated_generation_type_counts.get(gen_type, 0) + 1
            )

        # Count updated question types
        question_type = sample.get("type")
        if question_type:
            updated_question_type_counts[question_type] = (
                updated_question_type_counts.get(question_type, 0) + 1
            )

    print(f"Updated generation type distribution: {updated_generation_type_counts}")
    print(f"Updated question type distribution: {updated_question_type_counts}")

    # Update metadata
    data["metadata"]["generation_type_distribution"] = updated_generation_type_counts
    data["metadata"]["question_type_distribution"] = updated_question_type_counts

    # Save the updated JSON file
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(
        f"Successfully updated {entries_to_update} entries and saved changes to {filepath}"
    )


if __name__ == "__main__":
    dataset_path = "datasets/file_system_dataset.json"
    update_dataset_generated_types(dataset_path)
