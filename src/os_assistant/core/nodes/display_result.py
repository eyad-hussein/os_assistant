from datetime import datetime

from os_assistant.core.state import AssistantState
from os_assistant.pydantic_models.schemas import (
    CommandResponse,
    InformationResponse,
)


def display_result_node(state: AssistantState) -> AssistantState:
    """Display the final result to the user and record in conversation history"""
    print("\nNODE: display_result_node")

    if not state.get("final_result"):
        print("\nError: No final result generated.")
        return state

    final_result = state["final_result"]
    assert final_result is not None

    print("\n" + "=" * 60)
    print("LINUX ASSISTANT RESULT")
    print("=" * 60)

    print(f"Query: {final_result.query}")
    print(f"Domains analyzed: {', '.join(final_result.domains)}")

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

        print("\nCOMMAND FOR YOUR SYSTEM:")
        print(f"$ {command}")
        print("\nWHAT THIS COMMAND DOES:")
        print(what_command_does)
        if security_notes:
            print("\nSECURITY NOTES:")
            print(security_notes)

        if tool_breakdown:
            print("\nTOOL USAGE BREAKDOWN:")
            print(tool_breakdown)

        if tool_results:
            print("\nTOOL RESULTS:")
            print(tool_results)

        if tool_interpretation:
            print("\nTOOL RESULTS INTERPRETATION:")
            print(tool_interpretation)

    else:  # Information response
        # Validate structure before accessing keys
        assert type(response_data) is InformationResponse
        answer = response_data.answer
        sources = response_data.sources
        tool_breakdown = response_data.tool_breakdown
        tool_results = response_data.tool_results
        tool_interpretation = response_data.tool_interpretation

        print("\nABOUT YOUR SYSTEM:")
        print(answer)
        if sources:
            print("\nSOURCES FROM YOUR SYSTEM:")
            # Ensure sources is a list
            assert isinstance(sources, list)
            for source in sources:
                print(f"- {source}")

        if tool_breakdown:
            print("\nTOOL USAGE BREAKDOWN:")
            print(tool_breakdown)

        if tool_results:
            print("\nTOOL RESULTS:")
            print(tool_results)

        if tool_interpretation:
            print("\nTOOL RESULTS INTERPRETATION:")
            print(tool_interpretation)

    print("\n" + "=" * 60)

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
        print(f"Conversation history updated. Now contains {history_length} entries.")

    except Exception as e:
        print(f"Warning: Could not record conversation history: {e}")

    return state
