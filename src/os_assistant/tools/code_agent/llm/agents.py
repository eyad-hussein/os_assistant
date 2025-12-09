from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from os_assistant.utils.model_factory import coding_model, model
from os_assistant.utils.settings import CODE_AGENT  # Updated import

from ..core.models import CodeAnalysis, CodeExecutionState
from ..execution.executors import execute_code_in_subprocess
from ..llm.prompt_loader import (
    create_code_error_prompt,
    create_code_generation_prompt,
    create_summary_prompt,
)
from ..processing_utils.json_parsers import extract_json_manually
from ..processing_utils.markdown_parsers import extract_code_from_markdown
from ..processing_utils.string_utils import ensure_string
from ..processing_utils.structured_output_parsers import parse_structured_output


def code_executor_agent(state: CodeExecutionState) -> CodeExecutionState:
    """A node that executes code and updates the state."""
    llm = coding_model
    llm_summary = model

    # Extract code safely
    code = state.code
    if code:
        # Convert code to string if it's an AIMessage or similar object
        code_str = ensure_string(code)
        state.code = code_str

        # Create CodeAnalysis object for subprocess execution
        code_analysis = CodeAnalysis(
            code=code_str,
            dangerous=(
                state.danger_analysis.get("level", 1) if state.danger_analysis else 1
            ),
            reason=(
                state.danger_analysis.get("reason", "Default analysis")
                if state.danger_analysis
                else "Default analysis"
            ),
        )

        # Execute the code in subprocess
        code_result = execute_code_in_subprocess(code_analysis)

        # Update state with execution results
        state.execution_result = (
            code_result["stdout"] if code_result["stdout"] else "No output"
        )
        state.error_code = code_result["stderr"]

        # If there's an error, try to fix it
        if code_result["stderr"]:
            # Increment consecutive error counter
            state.consecutive_errors += 1

            # Abort if too many consecutive errors
            if state.consecutive_errors >= CODE_AGENT["MAX_CONSECUTIVE_ERRORS"]:
                return state

            # Ask LLM to fix the error
            error_prompt = create_code_error_prompt()
            error_response = llm.invoke(
                error_prompt.format(
                    question=ensure_string(state.question),
                    code=code_str,
                    error=code_result["stderr"],
                    output=(state.execution_result),
                )
            )

            try:
                parsed_result = parse_structured_output(error_response, CodeAnalysis)
                state.code = parsed_result.code
                state.danger_analysis = {
                    "level": parsed_result.dangerous,
                    "reason": parsed_result.reason,
                }
            except Exception:
                # Try manual JSON extraction if parsing fails
                json_data = extract_json_manually(error_response)
                if json_data and "code" in json_data:
                    state.code = json_data["code"]
                    state.danger_analysis = {
                        "level": json_data.get("dangerous", 1),
                        "reason": json_data.get(
                            "reason", "Extracted manually from response"
                        ),
                    }
                else:
                    # Fallback to simple code extraction if parsing fails
                    state.code = extract_code_from_markdown(error_response)
        else:
            # No errors, reset counter and generate summary
            state.consecutive_errors = 0
            summary_prompt = create_summary_prompt()

            # Generate summary with the execution result
            summary_response = llm_summary.invoke(
                summary_prompt.format(
                    code=code_str,
                    stdout=state.execution_result,
                )
            )

            # Store the complete summary
            state.agent_output = summary_response
    else:
        # Initial execution - generate and execute code
        generation_prompt = create_code_generation_prompt()
        response = llm.invoke(generation_prompt.format(instruction=state.question))

        try:
            # Parse the structured output
            parsed_result = parse_structured_output(response, CodeAnalysis)
            generated_code = parsed_result.code
            state.danger_analysis = {
                "level": parsed_result.dangerous,
                "reason": parsed_result.reason,
            }
        except Exception:
            # Try manual JSON extraction if parsing fails
            json_data = extract_json_manually(response)
            if json_data and "code" in json_data:
                generated_code = json_data["code"]
                state.danger_analysis = {
                    "level": json_data.get("dangerous", 1),
                    "reason": json_data.get(
                        "reason", "Extracted manually from response"
                    ),
                }
            else:
                # Fallback to basic code extraction if parsing fails
                generated_code = extract_code_from_markdown(response)
                state.danger_analysis = {
                    "level": 1,
                    "reason": "Parsing failed, default low risk assessment",
                }

        # Store the generated code
        state.code = generated_code

        # Create CodeAnalysis object for subprocess execution
        code_analysis = CodeAnalysis(
            code=generated_code,
            dangerous=(
                state.danger_analysis.get("level", 1) if state.danger_analysis else 1
            ),
            reason=(
                state.danger_analysis.get("reason", "Default analysis")
                if state.danger_analysis
                else "Default analysis"
            ),
        )

        # Execute in subprocess
        code_result = execute_code_in_subprocess(code_analysis)

        # Update state with execution results
        state.execution_result = (
            code_result["stdout"] if code_result["stdout"] else "No output"
        )
        state.error_code = code_result["stderr"]

        # If there's an error, prepare to rerun
        if code_result["stderr"]:
            state.consecutive_errors += 1
            error_prompt = create_code_error_prompt()
            error_response = llm.invoke(
                error_prompt.format(
                    question=ensure_string(state.question),
                    code=generated_code,
                    error=code_result["stderr"],
                    output=(state.execution_result),
                )
            )

            try:
                parsed_result = parse_structured_output(error_response, CodeAnalysis)
                state.code = parsed_result.code
                state.danger_analysis = {
                    "level": parsed_result.dangerous,
                    "reason": parsed_result.reason,
                }
            except Exception:
                # Try manual JSON extraction if parsing fails
                json_data = extract_json_manually(error_response)
                if json_data and "code" in json_data:
                    state.code = json_data["code"]
                    state.danger_analysis = {
                        "level": json_data.get("dangerous", 1),
                        "reason": json_data.get(
                            "reason", "Extracted manually from response"
                        ),
                    }
                else:
                    # Fallback to simple code extraction if parsing fails
                    state.code = extract_code_from_markdown(error_response)
        else:
            # No errors, reset counter and generate summary
            state.consecutive_errors = 0
            summary_prompt = create_summary_prompt()

            # Generate summary with the execution result
            summary_response = llm_summary.invoke(
                summary_prompt.format(
                    code=generated_code,
                    stdout=state.execution_result,
                )
            )

            # Store the complete summary
            state.agent_output = summary_response

    return state


def router(state: CodeExecutionState) -> str:
    """Determine next node based on state"""
    if state.consecutive_errors >= CODE_AGENT["MAX_CONSECUTIVE_ERRORS"]:
        return END
    if state.error_code and not state.agent_output:
        return "code_executor"
    return END


def create_code_execution_graph() -> CompiledStateGraph:
    """Create and configure the execution graph"""
    workflow = StateGraph(CodeExecutionState)
    workflow.add_node("code_executor", code_executor_agent)
    workflow.add_edge(START, "code_executor")
    workflow.add_conditional_edges("code_executor", router)
    return workflow.compile()
