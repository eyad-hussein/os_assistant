import io
import os
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, dict

from .config import TEMP_EXECUTION_FILE
from .models import CodeAnalysis


def execute_code_in_subprocess(code_analysis: CodeAnalysis) -> dict[str, Any]:
    """Execute code in a subprocess for isolation"""
    # Safety check - ask for confirmation if dangerous
    if code_analysis.dangerous == 3:
        print(f"\nWARNING: This operation has danger level {code_analysis.dangerous}/3")
        print(f"REASON: {code_analysis.reason}")
        print("\nGenerated code:")
        print(code_analysis.code)
        confirmation = input("Do you want to proceed? (y/n): ")
        if confirmation.lower() != "y":
            print("Operation cancelled by user.")
            return {"stdout": "Operation cancelled by user.", "stderr": None}

    # Execute the code
    print("\nExecuting code...")
    try:
        # Create a temporary Python file to execute
        with open(TEMP_EXECUTION_FILE, "w") as f:
            f.write(code_analysis.code)

        # Run the code and capture output
        result = subprocess.run(
            [sys.executable, TEMP_EXECUTION_FILE], capture_output=True, text=True
        )

        # Return results
        if result.returncode == 0:
            return {"stdout": result.stdout, "stderr": None}
        else:
            return {"stdout": result.stdout, "stderr": result.stderr}
    except Exception as e:
        return {"stdout": "", "stderr": f"Error executing code: {str(e)}"}
    finally:
        # Clean up temporary file
        try:
            if os.path.exists(TEMP_EXECUTION_FILE):
                os.remove(TEMP_EXECUTION_FILE)
        except Exception:
            pass


def execute_code_in_memory(
    code: str, danger_analysis: dict = None, interactive: bool = True
) -> dict[str, Any]:
    """Execute code in memory using exec()"""
    # Human-in-the-loop safety check
    if interactive and danger_analysis and danger_analysis.get("level", 0) >= 3:
        print(
            f"\nWARNING: This operation has danger level {danger_analysis['level']}/3"
        )
        print(f"REASON: {danger_analysis['reason']}")
        print("\nGenerated code:")
        print(code)

        while True:
            confirmation = input(
                "\nOptions:\n[y] Execute code\n[n] Cancel execution\n[e] Edit code\n[d] Show danger details\n[s] Run in isolated subprocess\nEnter choice: "
            )

            if confirmation.lower() == "y":
                print("Proceeding with execution...")
                break
            elif confirmation.lower() == "n":
                print("Operation cancelled by user.")
                return {"stdout": "Operation cancelled by user.", "stderr": None}
            elif confirmation.lower() == "e":
                print(
                    "\nEnter modified code (type 'DONE' on a new line when finished):"
                )
                new_code_lines = []
                while True:
                    line = input()
                    if line == "DONE":
                        break
                    new_code_lines.append(line)
                code = "\n".join(new_code_lines)
                print("\nCode updated.")
            elif confirmation.lower() == "d":
                print("\nDanger Assessment Details:")
                print(f"Level: {danger_analysis['level']}/3")
                print(f"Reasoning: {danger_analysis['reason']}")
                print("\nPotential risks of this type of operation:")
                if danger_analysis["level"] == 3:
                    print("- Could modify or delete important files")
                    print("- May execute unsafe system commands")
                    print("- Might access sensitive information")
                    print("- Could have unintended side effects")
            elif confirmation.lower() == "s":
                # Create a CodeAnalysis object for subprocess execution
                temp_analysis = CodeAnalysis(
                    code=code,
                    dangerous=danger_analysis.get("level", 3),
                    reason=danger_analysis.get(
                        "reason", "User requested isolated execution"
                    ),
                )
                return execute_code_in_subprocess(temp_analysis)
            else:
                print("Invalid option, please try again.")

    # Execute the code
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            # Create a local namespace for execution
            local_namespace = {}
            exec(code, {}, local_namespace)

        return {"stdout": stdout_buffer.getvalue(), "stderr": None}
    except Exception as e:
        return {
            "stdout": stdout_buffer.getvalue(),
            "stderr": f"{type(e).__name__}: {str(e)}\n{stderr_buffer.getvalue()}",
        }
