import io
import os
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

from ....utils.settings import CWD, TEMP_EXECUTION_FILE
from ..core.models import CodeAnalysis
from ..processing_utils.output_handler import (
    cleanup_temp_files,
    prepare_execution_environment,
)


def execute_code_in_subprocess(code_analysis: CodeAnalysis) -> dict[str, str | None]:
    """Execute code in a subprocess for isolation."""
    try:
        # Create a temporary Python file in the current working directory
        temp_file = os.path.join(CWD, TEMP_EXECUTION_FILE)
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(code_analysis.code)

        # Execute the file in a subprocess with proper environment variables
        env = os.environ.copy()
        env.update(prepare_execution_environment())

        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            cwd=CWD,
            env=env,
        )

        # Return results
        return {
            "stdout": result.stdout if result.returncode == 0 else None,
            "stderr": result.stderr if result.returncode != 0 else None,
        }
    except Exception as e:
        return {"stdout": None, "stderr": f"Error executing code: {str(e)}"}
    finally:
        cleanup_temp_files()


def execute_code_in_memory(code: str) -> dict[str, Any]:
    """Execute code in memory using exec()."""
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            exec(code, {"__builtins__": __builtins__})
        return {"stdout": stdout_buffer.getvalue(), "stderr": None}
    except Exception as e:
        return {
            "stdout": stdout_buffer.getvalue(),
            "stderr": f"{type(e).__name__}: {str(e)}",
        }
    finally:
        cleanup_temp_files()
