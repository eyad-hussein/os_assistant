import os
import tempfile
import time
from typing import Tuple, Optional, Dict, List

from ..config.config import OUTPUT_DIR, OUTPUT_FILE

# Constants for output file paths
STDOUT_FILE = os.path.join(OUTPUT_DIR, "code_agent_stdout.txt")
STDERR_FILE = os.path.join(OUTPUT_DIR, "code_agent_stderr.txt")

# Define a location for temporary output files
TEMP_OUTPUT_DIR = os.path.join(tempfile.gettempdir(), "os_assistant_outputs")
TEMP_OUTPUT_FILE = os.path.join(TEMP_OUTPUT_DIR, "execution_output.txt")
TEMP_RESULTS_FILE = os.path.join(TEMP_OUTPUT_DIR, "results.txt")


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
    os.makedirs(TEMP_OUTPUT_DIR, exist_ok=True)


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
        TEMP_OUTPUT_FILE,
        TEMP_RESULTS_FILE,
        "output.txt",
        "results.txt",
        "report.txt",
        "summary.txt",
    ]

    # First check in the temp directory
    for filename in common_output_files:
        # First check in the temp output directory
        temp_path = os.path.join(TEMP_OUTPUT_DIR, os.path.basename(filename))
        if os.path.exists(temp_path):
            try:
                with open(temp_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        captured_outputs.append(
                            f"--- Content of {os.path.basename(filename)} ---"
                        )
                        captured_outputs.append(content)
                        captured_outputs.append("---")
                        read_files.append(temp_path)
            except Exception as e:
                captured_outputs.append(f"Error reading {temp_path}: {str(e)}")

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
    if file_list:
        for file_path in file_list:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception:
                pass
    else:
        # Clean up standard temp files
        for filepath in [TEMP_OUTPUT_FILE, TEMP_RESULTS_FILE]:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except Exception:
                pass


def prepare_execution_environment() -> Dict[str, str]:
    """Prepare environment variables for execution

    Returns:
        Dict of environment variables to be added to execution environment
    """
    setup_output_directories()
    return {
        "TEMP_OUTPUT_FILE": TEMP_OUTPUT_FILE,
        "TEMP_RESULTS_FILE": TEMP_RESULTS_FILE,
        "TEMP_OUTPUT_DIR": TEMP_OUTPUT_DIR,
        "OUTPUT_FILE": OUTPUT_FILE,
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

            # Clean up the file after reading
            try:
                os.remove(OUTPUT_FILE)
            except Exception:
                pass

            return content
        except Exception as e:
            print(f"Error reading output file: {str(e)}")

    return None


def clear_output_file():
    """Remove the output file if it exists"""
    if os.path.exists(OUTPUT_FILE):
        try:
            os.remove(OUTPUT_FILE)
        except Exception:
            pass
