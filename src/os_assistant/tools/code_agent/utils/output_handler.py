import os

from ..config.config import CWD, TEMP_EXECUTION_FILE


def prepare_execution_environment() -> dict:
    """Prepare the execution environment with necessary variables"""
    # Set environment variables for code execution
    return {
        "CWD": CWD,
    }


def clear_output_file():
    """Placeholder for backward compatibility"""
    pass


def get_file_output() -> str:
    """Placeholder for backward compatibility"""
    return ""


def capture_file_outputs() -> tuple[str, list[str]]:
    """Placeholder for backward compatibility"""
    return "", []


def cleanup_temp_files():
    """Clean up any temporary files created during execution"""
    # Remove the temporary execution file
    temp_file = os.path.join(CWD, TEMP_EXECUTION_FILE)
    if os.path.exists(temp_file):
        try:
            os.remove(temp_file)
        except Exception as e:
            print(f"Warning: Could not remove temporary file {temp_file}: {str(e)}")
