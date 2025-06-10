from langchain_core.tools import tool

from .core.models import CodeAnalysis
from .core.run_code import run_code_execution
from .execution.executors import execute_code_in_subprocess
from .utils.output_handler import (
    clear_output_file,
    get_file_output,
    capture_file_outputs,
)
from .utils.parsers import ensure_string


@tool
def code_execute_tool(question: str) -> dict:
    """Tool to execute code based on a user's question.
    Args:
    question (str): The question or code to execute."""
    # Clear any previous output files before execution
    clear_output_file()

    # Run the code execution with the question
    tool_state = run_code_execution(question, verbose=True)

    # Check if execution was aborted due to too many errors
    if tool_state.get("execution_aborted", False):
        return {
            "question": question,
            "code": "Execution aborted due to too many consecutive errors.",
            "danger_analysis": {"level": 0, "reason": "Execution aborted"},
            "execution_result": "The code execution was stopped after 3 failed attempts.",
            "error_code": "Too many consecutive errors (3+).",
            "agent_output": "After multiple failed attempts, the execution was aborted for safety. Please try a different approach or simplify your request.",
        }

    # Get the code and danger analysis
    code = ensure_string(tool_state["code"])
    danger_analysis = tool_state["danger_analysis"] or {
        "level": 1,
        "reason": "Default analysis",
    }

    # Create a CodeAnalysis object for subprocess execution
    code_analysis = CodeAnalysis(
        code=code,
        dangerous=danger_analysis.get("level", 1),
        reason=danger_analysis.get("reason", "Default analysis"),
    )

    print("\nExecuting code in subprocess for more reliable output capture...")
    # Execute the code in subprocess (for better isolation and output capture)
    subprocess_result = execute_code_in_subprocess(code_analysis)

    # Use the subprocess output if available
    execution_result = (
        subprocess_result["stdout"]
        if subprocess_result["stdout"]
        else "No output from subprocess execution"
    )
    error_code = subprocess_result["stderr"]

    # Capture ALL file outputs and make sure they are included first in results
    file_output, _ = capture_file_outputs()
    if file_output:
        # Put file output FIRST for better visibility
        execution_result = (
            file_output + "\n\n" + execution_result if execution_result else file_output
        )

    # Prepare agent_output with full context including file output
    agent_output = tool_state.get("agent_output", "")
    if agent_output:
        agent_output = ensure_string(agent_output)

    # If we have file output but the agent didn't see it, regenerate the summary
    if file_output and file_output not in str(tool_state.get("execution_result", "")):
        print(
            "\nFile output found but not included in original summary. Using output with file content..."
        )
        from .llm.agents import create_llm, create_summary_prompt

        llm_summary = create_llm()
        summary_prompt = create_summary_prompt()

        new_summary = llm_summary.invoke(
            summary_prompt.format(
                code=code,
                stdout=execution_result,
            )
        )
        agent_output = ensure_string(new_summary)

    # Ensure all values are proper strings before returning
    return {
        "question": question,
        "code": code,
        "danger_analysis": danger_analysis,
        "execution_result": execution_result,
        "error_code": ensure_string(error_code) if error_code else None,
        "agent_output": agent_output,
    }


if __name__ == "__main__":
    result = code_execute_tool(
        "Which subdirectory in D:\\Graduation_Project_Test_Environment\\data contains the most .csv files? How many are there?"
    )
    print("\nTool execution completed successfully.")
    print(f"Result: {result['execution_result']}")
