import os
import time
import sys
from typing import Tuple, Optional, Dict, List

from ..config.config import OUTPUT_DIR, OUTPUT_FILE, RESULTS_FILE, CWD

# Constants for output file paths
STDOUT_FILE = os.path.join(OUTPUT_DIR, "code_agent_stdout.txt")
STDERR_FILE = os.path.join(OUTPUT_DIR, "code_agent_stderr.txt")


def setup_output_files():
    """Ensure output files exist and are empty"""
    # Make sure directories exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for file_path in [STDOUT_FILE, STDERR_FILE]:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("")


def setup_output_directories():
    """Ensure the output directories exist"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def read_output_files() -> Tuple[str, str]:
    """Read contents from output files"""
    stdout_content = ""
    stderr_content = ""

    try:
        if os.path.exists(STDOUT_FILE):
            with open(STDOUT_FILE, "r", encoding="utf-8") as f:
                stdout_content = f.read()
    except Exception as e:
        stderr_content += f"\nError reading stdout file: {str(e)}"

    try:
        if os.path.exists(STDERR_FILE):
            with open(STDERR_FILE, "r", encoding="utf-8") as f:
                stderr_content = f.read()
    except Exception as e:
        stderr_content += f"\nError reading stderr file: {str(e)}"

    return stdout_content, stderr_content


def capture_file_outputs() -> Tuple[str, List[str]]:
    """Check for and read any output files created during execution

    Returns:
        Tuple containing:
        - Combined output as a single string
        - List of file paths that were read
    """
    setup_output_directories()
    captured_outputs = []
    read_files = []

    # Check common output files
    common_output_files = [
        OUTPUT_FILE,
        RESULTS_FILE,
        "output.txt",
        "results.txt",
        "report.txt",
        "summary.txt",
    ]

    # Check files in the current working directory
    for filename in common_output_files:
        # First check in the output directory
        output_path = os.path.join(OUTPUT_DIR, os.path.basename(filename))
        if os.path.exists(output_path):
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        captured_outputs.append(
                            f"--- Content of {os.path.basename(filename)} ---"
                        )
                        captured_outputs.append(content)
                        captured_outputs.append("---")
                        read_files.append(output_path)
            except Exception as e:
                captured_outputs.append(f"Error reading {output_path}: {str(e)}")

        # Then check in the current directory
        if os.path.exists(filename):
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        captured_outputs.append(f"--- Content of {filename} ---")
                        captured_outputs.append(content)
                        captured_outputs.append("---")
                        read_files.append(filename)
            except Exception as e:
                captured_outputs.append(f"Error reading {filename}: {str(e)}")

    # Look for any new text files in the current directory (created during execution)
    for filename in os.listdir("."):
        if (
            filename.endswith(".txt")
            and os.path.isfile(filename)
            and filename not in common_output_files
        ):
            # Check if file was recently created (in the last minute)
            if os.path.getmtime(filename) > os.path.getmtime(__file__) - 60:
                try:
                    with open(filename, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if content:
                            captured_outputs.append(f"--- Content of {filename} ---")
                            captured_outputs.append(content)
                            captured_outputs.append("---")
                            read_files.append(filename)
                except Exception as e:
                    captured_outputs.append(f"Error reading {filename}: {str(e)}")

    return "\n".join(captured_outputs), read_files


def cleanup_temp_files(file_list: Optional[List[str]] = None):
    """Clean up temporary output files

    Args:
        file_list: List of specific files to clean up. If None, cleans up standard temp files.
    """
    print("\nCleaning up temporary files...")
    files_removed = 0

    # Remove the temp execution file
    temp_file = os.path.join(CWD, "temp_execution.py")
    if os.path.exists(temp_file):
        try:
            os.remove(temp_file)
            files_removed += 1
        except Exception as e:
            print(f"Could not remove {temp_file}: {e}")

    # Clean up temporary .txt files in several directories
    dirs_to_check = [
        CWD,  # Current working directory
        OUTPUT_DIR,  # Output directory
        os.path.join(CWD, "outputs"),  # Additional common output location
    ]

    # Check each directory for temp files
    for directory in dirs_to_check:
        if os.path.exists(directory) and os.path.isdir(directory):
            for filename in os.listdir(directory):
                if filename.endswith(".txt") and filename != "requirements.txt":
                    # Skip removing specific output files if needed
                    if filename in ["execution_output.txt", "results.txt"]:
                        continue

                    file_path = os.path.join(directory, filename)
                    try:
                        os.remove(file_path)
                        files_removed += 1
                    except Exception as e:
                        print(f"Could not remove {file_path}: {e}")

    # If specific files were provided, clean those up too
    if file_list:
        for file_path in file_list:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    files_removed += 1
                except Exception as e:
                    print(f"Could not remove {file_path}: {e}")

    if files_removed > 0:
        print(f"Removed {files_removed} temporary files")
    else:
        print("No temporary files found to clean up")


def prepare_execution_environment() -> Dict[str, str]:
    """Prepare environment variables for execution

    Returns:
        Dict of environment variables to be added to execution environment
    """
    setup_output_directories()
    return {
        "OUTPUT_DIR": OUTPUT_DIR,
        "OUTPUT_FILE": OUTPUT_FILE,
        "RESULTS_FILE": RESULTS_FILE,
        "CWD": CWD,
    }


def get_file_output(timeout: int = 2) -> Optional[str]:
    """
    Read the output from the file with a small timeout to ensure
    file operations are complete.

    Args:
        timeout: Number of seconds to wait for file operations to complete

    Returns:
        Contents of the output file or None if file doesn't exist
    """
    # Make sure directory exists
    setup_output_directories()

    # Small delay to ensure file operations complete
    time.sleep(timeout)

    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                content = f.read()
            return content
        except Exception as e:
            print(f"Error reading output file: {str(e)}")

    return None


def clear_output_file():
    """Clear the output file but don't delete it"""
    setup_output_directories()
    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write("")
    except Exception:
        pass


def log_progress(message, output_file=None):
    """Write a progress message both to stdout and to the output file immediately

    Args:
        message: The message to log
        output_file: Optional custom output file path. If None, uses environment variable
    """
    # Ensure the message ends with a newline
    if not message.endswith("\n"):
        message += "\n"

    # Print to stdout
    print(message, end="")
    sys.stdout.flush()

    # Write to the output file
    file_path = output_file or os.environ.get("OUTPUT_FILE", OUTPUT_FILE)
    try:
        # Make sure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Append mode to preserve previous messages
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(message)
            f.flush()  # Force immediate write to disk
    except Exception as e:
        print(f"Error writing to log file: {str(e)}")
