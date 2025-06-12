import os
import traceback

from ..llm.agents import create_code_execution_graph
from ..utils.output_handler import (
    cleanup_temp_files,
    get_file_output,
    clear_output_file,
)
from ..utils.parsers import ensure_string


def run_code_execution(question: str, verbose: bool = False, interactive: bool = True):
    """Run the code execution graph with the given question"""
    # Set interactive mode in environment for executors to access
    os.environ["INTERACTIVE_MODE"] = "1" if interactive else "0"

    try:
        # Clean up any leftover temp files from previous executions
        cleanup_temp_files()
        clear_output_file()

        # Initialize the graph
        code_execution_graph = create_code_execution_graph()

        # Set up the initial state
        initial_state = {"question": question}

        # Run the graph
        print(f"Processing question: {question}")
        print("=" * 50)

        final_state = code_execution_graph.invoke(initial_state)

        # Check if we hit the error limit (increase to 5)
        if final_state.get("consecutive_errors", 0) >= 5:
            print("\nExecution aborted: Too many consecutive errors (5+)")
            return {
                "question": question,
                "code": "",
                "danger_analysis": {"level": 0, "reason": "Execution aborted"},
                "execution_result": "",
                "error_code": "Too many consecutive errors. Execution aborted.",
                "agent_output": "After 5 consecutive failed attempts, execution was aborted for safety.",
                "execution_aborted": True,  # Flag to indicate execution was aborted
            }

        # Check for file output and prioritize it
        file_output = get_file_output()
        if file_output:
            execution_result = file_output
            if final_state.get("execution_result"):
                execution_result += "\n\n" + ensure_string(
                    final_state["execution_result"]
                )
            final_state["execution_result"] = execution_result

        # Print results
        if verbose:
            print("\nFull execution details:")
            print("-" * 50)
            print(f"Original question: {final_state['question']}")
            print("\nGenerated code:")
            print(f"```python\n{ensure_string(final_state['code'])}\n```")

            if final_state["danger_analysis"]:
                print("\nSafety analysis:")
                print(
                    f"Danger level: {final_state['danger_analysis'].get('level', 'Unknown')}/3"
                )
                print(
                    f"Reason: {final_state['danger_analysis'].get('reason', 'Not provided')}"
                )

            print("\nExecution output:")
            print(ensure_string(final_state["execution_result"]) or "No output")

            if final_state["error_code"]:
                print("\nErrors encountered:")
                print(ensure_string(final_state["error_code"]))

        if final_state["agent_output"]:
            print("\nFinal summary:")
            print("-" * 50)
            print(ensure_string(final_state["agent_output"]))

        return final_state

    except Exception as e:
        error_message = (
            f"Error during code execution: {str(e)}\n{traceback.format_exc()}"
        )
        print(error_message)
        return {
            "question": question,
            "code": "",
            "danger_analysis": {"level": 0, "reason": "Execution failed"},
            "execution_result": "",
            "error_code": error_message,
            "agent_output": f"The code execution process encountered an unexpected error: {str(e)}",
        }
    finally:
        # Clean up any temp files when done
        cleanup_temp_files()
