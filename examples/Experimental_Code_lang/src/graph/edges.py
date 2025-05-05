from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from os_assistant.graph.state import LinuxAssistantState


def check_domains_to_process(state: LinuxAssistantState) -> str:
    """Check if there are more domains to process for context retrieval"""
    if state.get("domains_to_process"):  # Check if list exists and is not empty
        print(f"Next domain for context: {state['domains_to_process'][0]}")
        return "context_retrieval_node"
    else:
        print(
            "All relevant domains processed for context. Moving to query classification."
        )
        return "query_classification_node"


def branch_on_query_type(state: LinuxAssistantState) -> str:
    """Branch based on query type"""
    query_type_result = state.get("query_type")
    if query_type_result is None:
        print("Query type missing, defaulting to information generation.")
        return "information_generation_node"
    match query_type_result.query_type:
        case "command":
            print("Query classified as command. Moving to command generation.")
            return "command_generation_node"
        case "information":
            print("Query classified as information. Moving to information generation.")
            return "information_generation_node"
