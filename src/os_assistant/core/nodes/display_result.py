from datetime import datetime

from os_assistant.core.state import AssistantState
from os_assistant.pydantic_models.schemas import (
    CommandResponse,
    InformationResponse,
)
from os_assistant.utils import LOGGER


def display_result_node(state: AssistantState) -> AssistantState:
    """Display the final result to the user and record in conversation history"""
    LOGGER.info("\nNODE: display_result_node")

    if not state.get("final_result"):
        LOGGER.error("No final result generated.")
        return state

    final_result = state["final_result"]
    assert final_result is not None

    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("LINUX ASSISTANT RESULT")
    LOGGER.info("=" * 60)

    LOGGER.info(f"Query: {final_result.query}")
    LOGGER.info(f"Domains analyzed: {', '.join(final_result.domains)}")

    response_data = final_result.response  # This is now always a dict

    if final_result.response_type == "command":
        # Validate structure before accessing keys
        assert type(response_data) is CommandResponse
        command = response_data.command
        what_command_does = response_data.what_command_does
        security_notes = response_data.security_notes
        tool_breakdown = response_data.tool_breakdown
        tool_results = response_data.tool_results
        tool_interpretation = response_data.tool_interpretation

        LOGGER.info("\nCOMMAND FOR YOUR SYSTEM:")
        LOGGER.info(f"$ {command}")
        LOGGER.info("\nWHAT THIS COMMAND DOES:")
        LOGGER.info(what_command_does)
        if security_notes:
            LOGGER.info("saved security notes")
            LOGGER.debug(f"\nSECURITY NOTES: {security_notes}")

        if tool_breakdown:
            LOGGER.info("saved tool usage breakdown")
            LOGGER.debug(f"\nTOOL USAGE BREAKDOWN: {tool_breakdown}")

        if tool_results:
            LOGGER.info("saved tool results")
            LOGGER.debug(f"\nTOOL RESULTS: {tool_results}")

        if tool_interpretation:
            LOGGER.info("saved tool results interpretation")
            LOGGER.debug(f"\nTOOL RESULTS INTERPRETATION: {tool_interpretation}")

    else:  # Information response
        # Validate structure before accessing keys
        assert type(response_data) is InformationResponse
        answer = response_data.answer
        sources = response_data.sources
        tool_breakdown = response_data.tool_breakdown
        tool_results = response_data.tool_results
        tool_interpretation = response_data.tool_interpretation

        LOGGER.info("\nABOUT YOUR SYSTEM:")
        LOGGER.info(answer)
        if sources:
            LOGGER.info("\nSOURCES FROM YOUR SYSTEM:")
            # Ensure sources is a list
            assert isinstance(sources, list)
            for source in sources:
                LOGGER.info(f"- {source}")

        if tool_breakdown:
            LOGGER.info("\nTOOL USAGE BREAKDOWN:")
            LOGGER.info(tool_breakdown)

        if tool_results:
            LOGGER.info("\nTOOL RESULTS:")
            LOGGER.info(tool_results)

        if tool_interpretation:
            LOGGER.info("\nTOOL RESULTS INTERPRETATION:")
            LOGGER.info(tool_interpretation)

    LOGGER.info("\n" + "=" * 60)

    # Record this interaction in conversation history
    try:
        # Create a conversation entry
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": state.get(
                "original_prompt", state["prompt"]
            ),  # Use original if available
            "refined_query": state["prompt"] if state.get("original_prompt") else None,
            "domains": final_result.domains,
            "response_type": final_result.response_type,
            "response": final_result.response,
        }

        # Initialize history if not present
        if "conversation_history" not in state:
            state["conversation_history"] = []

        # Add entry to history
        state["conversation_history"].append(entry)

        # Log the addition
        history_length = len(state["conversation_history"])
        LOGGER.info(
            f"Conversation history updated. Now contains {history_length} entries."
        )

    except Exception as e:
        LOGGER.warning(f"Could not record conversation history: {e}")

    return state
