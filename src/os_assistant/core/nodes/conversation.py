from os_assistant.utils.settings import (
    ASSISTANT_MODE,
    DOMAINS,
)
from os_assistant.core.nodes.helpers import get_mode_description
from os_assistant.core.state import AssistantState
from langchain.schema import HumanMessage
from os_assistant.utils.model_factory import model
from os_assistant.prompts.prompt_loader import load_prompt

def initialize_state(state: AssistantState, prompt: str) -> AssistantState:
    """Initialize the state with user prompt"""
    print("\nNODE: initialize_state")
    state["prompt"] = prompt
    state["domains"] = DOMAINS  # Use domains from config
    state["contexts"] = {}
    state["domains_to_process"] = []
    state["current_domain"] = None
    state["domain_analysis"] = None
    state["query_type"] = None
    state["command_response"] = None
    state["information_response"] = None
    state["final_result"] = None
    state["tool_usage_count"] = 0

    # Add assistant mode to state for debugging/logging
    state["assistant_mode"] = ASSISTANT_MODE
    print(f"Assistant Mode: {ASSISTANT_MODE} ({get_mode_description(ASSISTANT_MODE)})")

    return state


def conversation_context_node(state: AssistantState) -> AssistantState:
    """Provide conversation context by analyzing history and refining the prompt"""
    print("\nNODE: conversation_context_node")

    print("\nAnalyzing conversation context...")

    # Access conversation history
    conversation_history = state.get("conversation_history", [])

    # If this is the first interaction, nothing to enhance
    if not conversation_history:
        print("No conversation history found. Processing original query.")
        return state

    # Get the current prompt and previous interactions
    current_prompt = state["prompt"]

    # Format conversation history for the LLM with ranking by relevance
    formatted_history = ""

    # Include the most recent 3-5 interactions, prioritizing those that seem most relevant
    recent_history = conversation_history[-5:]
    for idx, entry in enumerate(recent_history):
        query = entry.get("query", "N/A")

        # Format the response based on the type
        response = entry.get("response", {})
        if isinstance(response, dict):
            if entry.get("response_type") == "command":
                cmd = response.get("command", "N/A")
                explanation = response.get("explanation", "N/A")
                formatted_history += f"Interaction {idx + 1}:\nUser: {query}\nAssistant: I suggested this command: '{cmd}'\n{explanation}\n\n"
            elif entry.get("response_type") == "information":
                answer = response.get("answer", "N/A")
                formatted_history += (
                    f"Interaction {idx + 1}:\nUser: {query}\nAssistant: {answer}\n\n"
                )
        else:
            formatted_history += (
                f"Interaction {idx + 1}:\nUser: {query}\nAssistant: {str(response)}\n\n"
            )

    # Load prompt from YAML
    conversation_context_yaml = load_prompt("conversation_context_node")

    # Format the prompt with required variables
    context_prompt = conversation_context_yaml["prompt"].format(
        formatted_history=formatted_history, current_prompt=current_prompt
    )

    # Ask the model to enhance the query
    messages = [HumanMessage(content=context_prompt)]
    model_response = model.invoke(messages)

    # Convert AIMessage to string properly, handling different response formats
    if hasattr(model_response, "content"):
        refined_prompt = str(model_response.content)
    else:
        refined_prompt = str(model_response)

    # Clean up any potential formatting issues
    refined_prompt = refined_prompt.strip()
    if refined_prompt.startswith('"') and refined_prompt.endswith('"'):
        refined_prompt = refined_prompt[1:-1]

    # If the model returns something that looks like an explanation rather than a query,
    # or if the refined prompt isn't substantially different, use the original
    print("INFO:", refined_prompt)
    if (
        "I don't need to enhance" in refined_prompt
        or "The query is self-contained" in refined_prompt
        or refined_prompt == current_prompt
    ):
        print("Query is self-contained or refinement unsuccessful. Using original.")
        return state

    print(f"Original query: {current_prompt}")
    print(f"Enhanced query: {refined_prompt}")

    # Store both the original and refined prompts
    state["original_prompt"] = current_prompt
    state["prompt"] = refined_prompt

    return state