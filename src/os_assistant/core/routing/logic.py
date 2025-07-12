from os_assistant.core.nodes.helpers import is_code_execution_enabled, is_rag_enabled
from os_assistant.core.nodes.registry import (
    COMMAND_NODE,
    CONTEXT_NODE,
    FINAL_NODE,
    INFO_NODE,
    QUERY_CLASS_NODE,
    TOOL_NODE,
)
from os_assistant.core.state import AssistantState


def check_domains_to_process(state: AssistantState) -> str:
    if not is_rag_enabled():
        print("RAG disabled. Skipping context retrieval.")
        return QUERY_CLASS_NODE

    domains_to_process = state.get("domains_to_process")
    if domains_to_process:
        print(f"Next domain for context: {domains_to_process[0]}")
        return CONTEXT_NODE

    print("All domains processed. Proceeding to query classification.")
    return QUERY_CLASS_NODE


def branch_on_query_type(state: AssistantState) -> str:
    query = state.get("query_type")
    if query is None:
        print("Query type missing. Defaulting to information generation.")
        return INFO_NODE

    if query.query_type == "command":
        print("Query classified as command.")
        return COMMAND_NODE
    else:
        print("Query classified as information.")
        return INFO_NODE


def check_for_tool_usage(state: AssistantState) -> str:
    if not is_code_execution_enabled():
        print("Tool disabled. Proceeding to final result.")
        return FINAL_NODE

    if state.get("tool_originating_node"):
        print("Tool usage detected. Routing to tool execution.")
        return TOOL_NODE

    print("No tool usage detected. Proceeding to final result.")
    return FINAL_NODE


def route_after_tool(state: AssistantState) -> str:
    origin = state.get("tool_originating_node")
    state["tool_originating_node"] = None
    if origin:
        print(f"Routing back to originating node: {origin}")
        return origin
    print("WARNING: No originating node found after tool. Going to final result.")
    return FINAL_NODE
