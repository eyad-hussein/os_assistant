from langchain.schema import HumanMessage

from os_assistant.core.state import AssistantState
from os_assistant.prompts.prompt_loader import load_prompt
from os_assistant.utils.model_factory import model


def conversation_context_node(state: AssistantState) -> AssistantState:
    """Provide conversation context by analyzing history and refining the prompt"""
    print("\nNODE: conversation_context_node")
    print("\nAnalyzing conversation context...")

    conversation_history = state.get("conversation_history", [])
    if not conversation_history:
        print("No conversation history found. Processing original query.")
        return state

    current_prompt = state["prompt"]

    # Include the most recent 3-5 interactions, prioritizing those that seem most relevant
    formatted_history = ""
    for idx, entry in enumerate(conversation_history[-5:]):
        query = entry.get("query", "N/A")
        response = entry.get("response", {})

        if isinstance(response, dict):
            if entry.get("response_type") == "command":
                cmd = response.get("command", "N/A")
                explanation = response.get("explanation", "N/A")
                formatted_history += (
                    f"Interaction {idx + 1}:\n"
                    f"User: {query}\n"
                    f"Assistant: I suggested this command: '{cmd}'\n{explanation}\n\n"
                )
            elif entry.get("response_type") == "information":
                answer = response.get("answer", "N/A")
                formatted_history += (
                    f"Interaction {idx + 1}:\nUser: {query}\nAssistant: {answer}\n\n"
                )
        else:
            formatted_history += (
                f"Interaction {idx + 1}:\nUser: {query}\nAssistant: {str(response)}\n\n"
            )

    # Build the conversation context prompt
    conversation_context_yaml = load_prompt("conversation_context_node")
    context_prompt = conversation_context_yaml["prompt"].format(
        formatted_history=formatted_history, current_prompt=current_prompt
    )

    # Invoke the model for prompt refinement
    messages = [HumanMessage(content=context_prompt)]
    model_response = model.invoke(messages)

    refined_prompt = getattr(model_response, "content", str(model_response)).strip()
    if refined_prompt.startswith('"') and refined_prompt.endswith('"'):
        refined_prompt = refined_prompt[1:-1]

    # TODO: Replace this heuristic with a structured flag returned from the model
    # as prompt_is_refined: ture/false

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
