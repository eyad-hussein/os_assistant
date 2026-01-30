from dagent.core.nodes.helpers import is_code_execution_enabled, is_rag_enabled
from dagent.core.nodes.registry import (
    COMMAND_NODE,
    CONTEXT_NODE,
    FINAL_NODE,
    INFO_NODE,
    QUERY_CLASS_NODE,
    TOOL_NODE,
)
from dagent.core.state import AssistantState
from dagent.utils import LOGGER


def check_domains_to_process(state: AssistantState) -> str:
    if not is_rag_enabled():
        LOGGER.warning("RAG disabled. Skipping context retrieval.")
        return QUERY_CLASS_NODE

    domains_to_process = state.get("domains_to_process")
    if domains_to_process:
        LOGGER.info(f"Next domain for context: {domains_to_process[0]}")
        return CONTEXT_NODE

    LOGGER.info("All domains processed. Proceeding to query classification.")
    return QUERY_CLASS_NODE


def branch_on_query_type(state: AssistantState) -> str:
    query = state.get("query_type")
    if query is None:
        LOGGER.error("Query type missing. Defaulting to information generation.")
        return INFO_NODE

    if query.query_type == "command":
        LOGGER.info("Query classified as command.")
        return COMMAND_NODE
    else:
        LOGGER.info("Query classified as information.")
        return INFO_NODE


def check_for_tool_usage(state: AssistantState) -> str:
    if not is_code_execution_enabled():
        LOGGER.warning("Tool disabled. Proceeding to final result.")
        return FINAL_NODE

    if state.get("tool_originating_node"):
        LOGGER.info("Tool usage detected. Routing to tool execution.")
        return TOOL_NODE

    LOGGER.warning("No tool usage detected. Proceeding to final result.")
    return FINAL_NODE


def route_after_tool(state: AssistantState) -> str:
    origin = state.get("tool_originating_node")
    state["tool_originating_node"] = None
    if origin:
        LOGGER.info(f"Routing back to originating node: {origin}")
        return origin
    LOGGER.warning("No originating node found after tool. Going to final result.")
    return FINAL_NODE
