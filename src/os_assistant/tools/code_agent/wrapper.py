from langchain_core.tools import tool

from .core.models import CodeAnalysis
from .core.run_code import run_code_execution
from .execution.executors import execute_code_in_subprocess
from .utils.parsers import ensure_string


@tool
def code_execute_tool(question: str) -> dict:
    """Tool to execute code based on a user's question.
    Args:
    question (str): The question or code to execute."""

    try:
        # Run the code execution with the question
        tool_state = run_code_execution(question, verbose=False)

        # Check if execution was aborted due to too many errors
        if tool_state.get("execution_aborted", False):
            return {
                "question": question,
                "code": "Execution aborted due to too many consecutive errors.",
                "danger_analysis": {"level": 1, "reason": "Execution aborted"},
                "execution_result": "The code execution was stopped after 5 failed attempts.",
                "error_code": "Too many consecutive errors (5+).",
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

        # Execute the code in subprocess
        subprocess_result = execute_code_in_subprocess(code_analysis)

        # Use the subprocess output if available
        execution_result = (
            subprocess_result["stdout"] or "No output from subprocess execution"
        )
        error_code = subprocess_result["stderr"]

        # Prepare agent_output
        agent_output = ensure_string(tool_state.get("agent_output", ""))

        # Ensure all values are proper strings before returning
        return {
            "question": question,
            "code": code,
            "danger_analysis": danger_analysis,
            "execution_result": execution_result,
            "error_code": ensure_string(error_code) if error_code else None,
            "agent_output": agent_output,
        }
    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        return {
            "question": question,
            "code": "",
            "danger_analysis": {"level": 0, "reason": "Execution error"},
            "execution_result": f"Error in code execution tool: {str(e)}",
            "error_code": error_details,
            "agent_output": "The code execution encountered an error. Please try a different approach.",
        }


if __name__ == "__main__":
    result = code_execute_tool("what is my current working directory?")
    print(f"Result: {result['execution_result']}")
