from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from ..config.config import (
    LLM_MODEL,
    LLM_MODEL_CODING,
    LLM_TEMPERATURE,
    OLLAMA_BASE_URL,
)
from ..core.models import CodeAnalysis, CodeExecutionState
from ..execution.executors import execute_code_in_memory, execute_code_in_subprocess
from ..llm.prompts import (
    create_code_error_prompt,
    create_code_generation_prompt,
    create_summary_prompt,
)
from ..utils.output_handler import capture_file_outputs, get_file_output
from ..utils.parsers import (
    ensure_string,
    extract_code_from_markdown,
    parse_structured_output,
    extract_json_manually,
)


def create_llm_coding():
    """Create and configure the LLM"""
    return ChatOllama(
        model=LLM_MODEL_CODING, temperature=LLM_TEMPERATURE, base_url=OLLAMA_BASE_URL
    )


def create_llm():
    """Create and configure the LLM"""
    return ChatOllama(
        model=LLM_MODEL, temperature=LLM_TEMPERATURE, base_url=OLLAMA_BASE_URL
    )


def code_executor_agent(state: CodeExecutionState) -> CodeExecutionState:
    """
    A node that executes code and updates the state.
    """
    llm = create_llm_coding()
    llm_summary = create_llm()
    # Extract code safely
    code = state.code
    if code:
        # Convert code to string if it's an AIMessage or similar object
        code_str = ensure_string(code)

        # Update state with the string version for safer usage
        state.code = code_str

        print("Code to execute:")
        print(code_str)
        print("End the code")

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

        # Execute the code in subprocess for better output capturing
        print("Executing code in subprocess for better output capturing...")
        code_result = execute_code_in_subprocess(code_analysis)

        # Update state with execution results from subprocess
        state.execution_result = (
            code_result["stdout"] if code_result["stdout"] else "No output"
        )
        state.error_code = code_result["stderr"]

        # Check for file outputs and add them to the result
        file_output, _ = capture_file_outputs()
        if file_output:
            print(f"Found file output: {file_output}")
            # Add file output to execution result
            if state.execution_result and state.execution_result != "No output":
                state.execution_result = f"{state.execution_result}\n\n{file_output}"
            else:
                state.execution_result = file_output

        # If there's an error, update question to include error info and return to try again
        if code_result["stderr"]:
            print(f"Encountered error: {code_result['stderr']}")

            # Increment consecutive error counter
            state.consecutive_errors += 1
            print(f"Consecutive errors: {state.consecutive_errors}")

            # Check if we should abort due to too many errors - Update from 3 to 5
            if state.consecutive_errors >= 5:
                print("Too many consecutive errors (5+). Aborting execution.")
                return state

            print("Asking LLM to fix the error...")

            error_prompt = create_code_error_prompt()
            error_response = llm.invoke(
                error_prompt.format(
                    question=ensure_string(state.question),
                    code=code_str,  # Use the string version
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
                print(
                    f"Generated fixed code with danger level: {parsed_result.dangerous}"
                )
            except Exception as e:
                print(f"Error parsing LLM response: {str(e)}")
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
                    print("Successfully extracted code using manual JSON extraction")
                else:
                    # Fallback to simple code extraction if parsing fails
                    state.code = extract_code_from_markdown(error_response)
        else:
            # No errors, reset counter and generate summary
            state.consecutive_errors = 0
            summary_prompt = create_summary_prompt()

            # Pass the full output including file output to the summary agent
            complete_output = state.execution_result

            # Double-check for any additional file output
            additional_file_output = get_file_output()
            if additional_file_output and additional_file_output not in complete_output:
                complete_output = f"{complete_output}\n\n{additional_file_output}"

            print("\nGenerating summary with complete output:")
            print("-" * 50)
            print(complete_output)
            print("-" * 50)

            summary_response = llm_summary.invoke(
                summary_prompt.format(
                    code=code_str,
                    stdout=complete_output,
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

            # Store danger analysis
            state.danger_analysis = {
                "level": parsed_result.dangerous,
                "reason": parsed_result.reason,
            }
        except Exception as e:
            print(f"Error parsing LLM response during code generation: {str(e)}")
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
                print("Successfully extracted code using manual JSON extraction")
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

        # Execute in subprocess for better output capturing
        print("Executing generated code in subprocess...")
        code_result = execute_code_in_subprocess(code_analysis)

        # Update state with execution results
        state.execution_result = (
            code_result["stdout"] if code_result["stdout"] else "No output"
        )
        state.error_code = code_result["stderr"]

        # Check for file outputs and add them to the result
        file_output, _ = capture_file_outputs()
        if file_output:
            print(f"Found file output: {file_output}")
            # Add file output to execution result
            if state.execution_result and state.execution_result != "No output":
                state.execution_result = f"{state.execution_result}\n\n{file_output}"
            else:
                state.execution_result = file_output

        # If there's an error, prepare to rerun
        if code_result["stderr"]:
            # Increment consecutive error counter
            state.consecutive_errors += 1
            print(f"Consecutive errors: {state.consecutive_errors}")

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
            except Exception as e:
                print(f"Error parsing LLM response: {str(e)}")
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
                    print("Successfully extracted code using manual JSON extraction")
                else:
                    # Fallback to simple code extraction if parsing fails
                    state.code = extract_code_from_markdown(error_response)
        else:
            # No errors, reset counter and generate summary
            state.consecutive_errors = 0
            summary_prompt = create_summary_prompt()

            # Pass the full output including file output to the summary agent
            complete_output = state.execution_result

            # Double-check for any additional file output
            additional_file_output = get_file_output()
            if additional_file_output and additional_file_output not in complete_output:
                complete_output = f"{complete_output}\n\n{additional_file_output}"

            print("\nGenerating summary with complete output:")
            print("-" * 50)
            print(complete_output)
            print("-" * 50)

            summary_response = llm_summary.invoke(
                summary_prompt.format(
                    code=generated_code,
                    stdout=complete_output,
                )
            )

            # Store the complete summary
            state.agent_output = summary_response

    return state


def router(state: CodeExecutionState):
    """Determine next node based on state"""
    # If we hit the error limit, end execution
    if state.consecutive_errors >= 5:  # Increase from 3 to 5
        return END

    # If there's an error and no final output, we need to loop back
    if state.error_code and not state.agent_output:
        return "code_executor"
    # Otherwise, we're done
    return END


def create_code_execution_graph():
    """Create and configure the execution graph"""
    workflow = StateGraph(CodeExecutionState)
    workflow.add_node("code_executor", code_executor_agent)

    # Connect the nodes with conditional routing
    workflow.add_edge(START, "code_executor")
    workflow.add_conditional_edges("code_executor", router)

    # Compile the graph
    return workflow.compile()
