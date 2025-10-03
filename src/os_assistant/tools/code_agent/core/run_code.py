import os
import traceback

from os_assistant.utils import LOGGER

from ....configs import CODE_AGENT  # Import the configuration
from ..llm.agents import create_code_execution_graph
from ..processing_utils.output_handler import cleanup_temp_files
from ..processing_utils.string_utils import ensure_string


def run_code_execution(question: str, verbose: bool = False, interactive: bool = True):
    """Run the code execution graph with the given question"""
    # Set interactive mode in environment for executors to access
    os.environ["INTERACTIVE_MODE"] = "1" if interactive else "0"

    try:
        # Clean up any leftover temp files from previous executions
        cleanup_temp_files()

        # Initialize the graph
        code_execution_graph = create_code_execution_graph()

        # Set up the initial state
        initial_state = {"question": question}

        # Run the graph
        if verbose:
            LOGGER.info(f"Processing question: {question}")
            LOGGER.info("=" * 50)

        final_state = code_execution_graph.invoke(initial_state)

        # Check if we hit the error limit
        if (
            final_state.get("consecutive_errors", 0)
            >= CODE_AGENT["MAX_CONSECUTIVE_ERRORS"]
        ):
            if verbose:
                LOGGER.error("\nExecution aborted: Too many consecutive errors (5+)")
            return {
                "question": question,
                "code": "",
                "danger_analysis": {"level": 0, "reason": "Execution aborted"},
                "execution_result": "",
                "error_code": f"Too many consecutive errors. Execution aborted after {CODE_AGENT['MAX_CONSECUTIVE_ERRORS']} attempts.",
                "agent_output": "After 5 consecutive failed attempts, execution was aborted for safety.",
                "execution_aborted": True,
            }

        # Print results if verbose
        if verbose:
            LOGGER.info("\nFull execution details:")
            LOGGER.info("-" * 50)
            LOGGER.info(f"Original question: {final_state['question']}")
            LOGGER.info("\nGenerated code:")
            LOGGER.info(f"```python\n{ensure_string(final_state['code'])}\n```")

            if final_state["danger_analysis"]:
                LOGGER.debug("\nSafety analysis:")
                LOGGER.debug(
                    f"Danger level: {final_state['danger_analysis'].get('level', 'Unknown')}/3"
                )
                LOGGER.debug(
                    f"Reason: {final_state['danger_analysis'].get('reason', 'Not provided')}"
                )

            LOGGER.info("\nExecution output:")
            LOGGER.info(ensure_string(final_state["execution_result"]) or "No output")

            if final_state["error_code"]:
                LOGGER.error(
                    f"\nErrors encountered:\n{ensure_string(final_state['error_code'])}"
                )

            if final_state["agent_output"]:
                LOGGER.info(
                    f"\nFinal summary:\n{'-' * 50}\n{ensure_string(final_state['agent_output'])}"
                )

        return final_state

    except Exception as e:
        error_message = (
            f"Error during code execution: {str(e)}\n{traceback.format_exc()}"
        )
        if verbose:
            LOGGER.error(error_message)
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
