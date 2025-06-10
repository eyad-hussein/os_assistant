from langchain_core.tools import tool

from .core.run_code import run_code_execution
from .utils.parsers import ensure_string


@tool
def code_execute_tool(question: str) -> dict:
    """Tool to execute code based on a user's question.
    Args:
    question (str): The question or code to execute."""
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

    # Ensure execution_result is complete and preserve all information
    full_execution_result = ensure_string(tool_state["execution_result"])

    # Prepare agent_output with explicit instructions to be complete
    agent_output = tool_state.get("agent_output", "")
    if agent_output:
        agent_output = ensure_string(agent_output)

    # Ensure all values are proper strings before returning
    return {
        "question": question,
        "code": ensure_string(tool_state["code"]),
        "danger_analysis": tool_state["danger_analysis"],
        "execution_result": full_execution_result,
        "error_code": (
            ensure_string(tool_state["error_code"])
            if tool_state["error_code"]
            else None
        ),
        "agent_output": agent_output,
    }


if __name__ == "__main__":
    result = code_execute_tool(
        " What is the longest directory name within my blabla directory?"
    )
    print("\nTool execution completed successfully.")
    print(f"Result: {result['agent_output']}")
