from os_assistant.core.state import AssistantState
from os_assistant.pydantic_models.schemas import (
    FinalResult,
    InformationResponse,
)
from os_assistant.utils import LOGGER


def add_mode_note_to_response(response, is_code_execution_enabled):
    """Add a mode-specific note to a response if needed"""
    if not is_code_execution_enabled and "disabled in the current mode" not in response:
        return (
            response
            + "\n\nNote: This response was generated without using the code execution tool, which is disabled in the current mode. It is based on general knowledge."
        )
    return response


def prepare_final_result_node(state: AssistantState) -> AssistantState:
    """Prepare the final result"""
    LOGGER.info("\nNODE: prepare_final_result_node")

    # Ensure domain_analysis and query_type exist before accessing keys
    domains_tmp = state.get("domain_analysis")
    if domains_tmp is None:
        LOGGER.warning("Domain analysis missing, using all domains for final result.")
        domains = state["domains"]  # Fallback to all domains
    else:
        domains = domains_tmp.domains

    query_type_tmp = state.get("query_type")
    if query_type_tmp is None:
        LOGGER.warning(
            "Query type missing, defaulting to 'information' for final result."
        )
        response_type = "information"  # Fallback type
    else:
        response_type = query_type_tmp.query_type

    # Create context summary
    context_summary = "Analyzed information from: "
    context_summary += ", ".join(domains)

    # Determine response content
    match response_type:
        case "command":
            if state.get("command_response"):
                response = state["command_response"]
            else:
                LOGGER.warning("Command response expected but missing.")
                # Create a fallback command response if needed, or switch type
                response_type = "information"  # Switch to info if command failed
                response = InformationResponse(
                    answer=f"Could not generate a command for '{state['prompt']}'. Please try rephrasing.",
                    sources=["System processing error"],
                )
        # Handle information response (either primary or fallback)
        case "information":
            if state.get("information_response"):
                response = state["information_response"]
            else:
                LOGGER.warning("Information response expected but missing.")
                # Create a fallback information response
                response = InformationResponse(
                    answer=f"Unable to generate an answer for '{state['prompt']}' based on the available information.",
                    sources=["System processing error"],
                )

    # Ensure response is not None before creating FinalResult
    if response is None:
        LOGGER.error("Could not determine a valid response for the final result.")
        # Handle this case, maybe set final_result to an error state or raise exception
        # For now, create a minimal error response
        response = InformationResponse(
            answer="An unexpected error occurred while generating the response.",
            sources=["System error"],
        )
        response_type = "information"  # Ensure type matches the fallback

    # Create final result
    final_result = FinalResult(
        query=state["prompt"],
        domains=domains,
        response_type=response_type,  # Use the potentially updated response_type
        response=response,  # Pass the dictionary directly
        context_summary=context_summary,
    )

    state["final_result"] = final_result

    return state
