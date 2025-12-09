import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

from os_assistant.utils import LOGGER
from os_assistant.utils.settings import CWD, TEMP_EXECUTION_FILE

from ..core.models import CodeAnalysis
from ..processing_utils.json_parsers import extract_json_manually
from ..processing_utils.output_handler import (
    cleanup_temp_files,
    prepare_execution_environment,
)
from ..processing_utils.string_utils import ensure_string


def execute_code_in_subprocess(code_analysis: CodeAnalysis) -> dict[str, str | None]:
    """Execute code in a subprocess for isolation"""
    # Safety check - ask for confirmation if dangerous
    if code_analysis.dangerous == 3:
        LOGGER.warning(
            f"\nThis operation has danger level {code_analysis.dangerous}/3\n"
            f"REASON: {code_analysis.reason}"
        )
        LOGGER.info(f"Generated code:\n{code_analysis.code}")
        confirmation = input("Do you want to proceed? (y/n): ")
        if confirmation.lower() != "y":
            LOGGER.info("Operation cancelled by user.")
            return {"stdout": "Operation cancelled by user.", "stderr": None}

    try:
        # Create a temporary Python file in the current working directory
        temp_file = os.path.join(CWD, TEMP_EXECUTION_FILE)
        with open(temp_file, "w", encoding="utf-8") as f:
            # Add essential imports
            f.write("import os\nimport sys\nimport io\nimport traceback\n\n")

            # Fix common syntax errors in f-strings before writing
            fixed_code = code_analysis.code
            fixed_code = fixed_code.replace('\n")', '")')
            fixed_code = fixed_code.replace('\\n")', '")')

            # Look for common f-string errors
            if "f'" in fixed_code or 'f"' in fixed_code:
                lines = fixed_code.split("\n")
                fixed_lines = []
                for line in lines:
                    if '\\n")' in line or '\n")' in line:
                        line = line.replace('\\n")', '")')
                        line = line.replace('\n")', '")')
                    fixed_lines.append(line)
                fixed_code = "\n".join(fixed_lines)

            # Write the fixed code to the temp file
            f.write(fixed_code)

        # Execute the file in a subprocess with proper environment variables
        env = os.environ.copy()
        env.update(prepare_execution_environment())

        # Run the subprocess in the current working directory
        result = subprocess.run(
            [sys.executable, temp_file],
            capture_output=True,
            text=True,
            cwd=CWD,
            env=env,
        )

        # Return results
        if result.returncode == 0:
            return {"stdout": result.stdout, "stderr": None}
        else:
            return {"stdout": result.stdout, "stderr": result.stderr}
    except Exception as e:
        error_msg = f"Error executing code: {str(e)}"
        return {"stdout": "", "stderr": error_msg}
    finally:
        # Clean up temporary file
        cleanup_temp_files()


def execute_code_in_memory(
    code: Any, danger_analysis: dict | None = None, interactive: bool = True
) -> dict[str, Any]:
    """Execute code in memory using exec()"""
    # Convert code to string if it's an AIMessage or similar
    code = ensure_string(code)

    # Extract code from JSON if needed
    extracted_code = None
    try:
        stripped_code = code.strip()
        if stripped_code.startswith("{"):
            json_data = json.loads(stripped_code)
            if isinstance(json_data, dict) and "code" in json_data:
                if (
                    isinstance(json_data["code"], dict)
                    and "python_code" in json_data["code"]
                ):
                    extracted_code = json_data["code"]["python_code"]
                else:
                    extracted_code = json_data["code"]
        elif "json" in stripped_code[:10].lower() and "{" in stripped_code:
            json_start = stripped_code.find("{")
            if json_start >= 0:
                json_data = extract_json_manually(stripped_code[json_start:])
                if json_data and "code" in json_data:
                    extracted_code = json_data["code"]
    except Exception:
        pass

    # If JSON extraction succeeded, use the extracted code
    if extracted_code is not None:
        code = extracted_code
    # Otherwise, try markdown extraction
    elif "```python" in code and "```" in code:
        try:
            code_blocks = code.split("```python")[1:]
            for block in code_blocks:
                if "```" in block:
                    code = block.split("```")[0].strip()
                    break
        except Exception:
            pass

    # Human-in-the-loop safety check
    if interactive and danger_analysis and danger_analysis.get("level", 0) >= 3:
        LOGGER.warning(
            f"\nThis operation has danger level {danger_analysis['level']}/3\n"
            f"REASON: {danger_analysis['reason']}"
        )
        LOGGER.info(f"Generated code:\n{code}")

        while True:
            confirmation = input(
                "\nOptions:\n[y] Execute code\n[n] Cancel execution\n[e] Edit code\n[d] Show danger details\n[s] Run in isolated subprocess\nEnter choice: "
            )

            if confirmation.lower() == "y":
                break
            elif confirmation.lower() == "n":
                return {"stdout": "Operation cancelled by user.", "stderr": None}
            elif confirmation.lower() == "e":
                LOGGER.info(
                    "\nEnter modified code (type 'DONE' on a new line when finished):"
                )
                new_code_lines = []
                while True:
                    line = input()
                    if line == "DONE":
                        break
                    new_code_lines.append(line)
                code = "\n".join(new_code_lines)
                LOGGER.info("\nCode updated.")
            elif confirmation.lower() == "d":
                LOGGER.info(
                    "\nDanger Assessment Report:\n"
                    f"Level: {danger_analysis['level']}/3\n"
                    f"Reasoning: {danger_analysis['reason']}"
                )
                if danger_analysis["level"] == 3:
                    LOGGER.info(
                        "\nPotential risks of this type of operation:\n"
                        "- Could modify or delete important files\n"
                        "- May execute unsafe system commands\n"
                        "- Might access sensitive information\n"
                        "- Could have unintended side effects"
                    )
            elif confirmation.lower() == "s":
                temp_analysis = CodeAnalysis(
                    code=code,
                    dangerous=danger_analysis.get("level", 3),
                    reason=danger_analysis.get(
                        "reason", "User requested isolated execution"
                    ),
                )
                return execute_code_in_subprocess(temp_analysis)
            else:
                LOGGER.error("Invalid option, please try again.")

    # Execute the code
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            globals_dict = {
                "os": os,
                "sys": sys,
                "subprocess": subprocess,
                "__builtins__": __builtins__,
            }

            # Add environment variables for execution
            env_vars = prepare_execution_environment()
            for key, value in env_vars.items():
                globals_dict[key] = value

            exec(code, globals_dict)

        return {"stdout": stdout_buffer.getvalue(), "stderr": None}
    except Exception as e:
        return {
            "stdout": stdout_buffer.getvalue(),
            "stderr": f"{type(e).__name__}: {str(e)}\n{stderr_buffer.getvalue()}",
        }
    finally:
        cleanup_temp_files()
