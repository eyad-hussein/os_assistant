from langchain_core.tools import tool

from .run_code import run_code_execution
from .parsers import ensure_string


@tool
def code_execute_tool(question: str) -> dict:
    """Tool to execute code based on a user's question.
    Args:
    question (str): The question or code to execute."""
    tool_state = run_code_execution(question, verbose=True)

    # Ensure all values are proper strings before returning
    return {
        "question": question,
        "code": ensure_string(tool_state["code"]),
        "danger_analysis": tool_state["danger_analysis"],
        "execution_result": ensure_string(tool_state["execution_result"]),
        "error_code": (
            ensure_string(tool_state["error_code"])
            if tool_state["error_code"]
            else None
        ),
        "agent_output": (
            ensure_string(tool_state["agent_output"])
            if tool_state["agent_output"]
            else None
        ),
    }


if __name__ == "__main__":
    result = code_execute_tool(
        " What is the longest directory name within my current working directory?"
    )
    print("\nTool execution completed successfully.")
    print(f"Result: {result['agent_output']}")
