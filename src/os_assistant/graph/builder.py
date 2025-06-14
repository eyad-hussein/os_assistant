from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from os_assistant.config.settings import ASSISTANT_MODE
from os_assistant.graph.nodes import (
    command_generator_node,
    context_retrieval_node,
    conversation_context_node,
    display_result_node,
    domain_analysis_node,
    information_generator_node,
    prepare_final_result_node,
    query_classifier_node,
    tool_execution_node,
)
from os_assistant.graph.state import LinuxAssistantState

# --- Helper functions for edge conditions ---


def is_rag_enabled():
    """Check if RAG is enabled in the current mode"""
    return ASSISTANT_MODE in [0, 2]


def is_tool_enabled():
    """Check if code tool is enabled in the current mode"""
    return ASSISTANT_MODE in [0, 1]


# --- Edge Functions ----


def check_for_tool_usage(state: LinuxAssistantState) -> str:
    """Check if we need to route to tool execution"""
    # Skip tool execution completely if tool is disabled
    if not is_tool_enabled():
        print("Tool disabled in current mode. Proceeding to final result.")
        return "prepare_final_result_node"

    # Check if tool was requested
    if state.get("tool_originating_node") is not None:
        print("Tool usage detected. Routing to tool execution.")
        return "tool_execution_node"

    print("No tool usage detected. Proceeding to final result.")
    return "prepare_final_result_node"


def route_after_tool(state: LinuxAssistantState) -> str:
    """Route back to originating node after tool execution"""
    # Get originating node and clear it to prevent loops
    originating_node = state.get("tool_originating_node")
    if originating_node:
        state["tool_originating_node"] = None

    print(f"Routing after tool execution. Originating node: {originating_node}")

    # Return to appropriate node
    if originating_node == "command_generation_node":
        return "command_generation_node"
    elif originating_node == "information_generation_node":
        return "information_generation_node"

    # Default fallback path - should not reach here if routing is correct
    print("WARNING: No originating node found after tool execution. Using fallback.")
    return "prepare_final_result_node"


def check_domains_to_process(state: LinuxAssistantState) -> str:
    """Check if there are more domains to process for context retrieval"""
    # Skip context retrieval if RAG is disabled
    if not is_rag_enabled():
        print("RAG disabled in current mode. Skipping context retrieval.")
        return "query_classification_node"

    # Check if there are more domains to process
    if state.get("domains_to_process"):
        print(f"Next domain for context: {state['domains_to_process'][0]}")
        return "context_retrieval_node"

    print("All relevant domains processed for context. Moving to query classification.")
    return "query_classification_node"


def branch_on_query_type(state: LinuxAssistantState) -> str:
    """Branch based on query type"""
    query_type_result = state.get("query_type")

    # Default to information if missing
    if query_type_result is None:
        print("Query type missing, defaulting to information generation.")
        return "information_generation_node"

    # Branch based on query type
    if query_type_result.query_type == "command":
        print("Query classified as command. Moving to command generation.")
        return "command_generation_node"
    else:
        print("Query classified as information. Moving to information generation.")
        return "information_generation_node"


# --- Build the Graph ---


def build_linux_assistant_graph():
    """Build the LangGraph for the Linux assistant"""
    # Create a new graph
    workflow = StateGraph(LinuxAssistantState)

    # Add all nodes to the graph
    workflow.add_node("conversation_context_node", conversation_context_node)
    workflow.add_node("domain_analysis_node", domain_analysis_node)
    workflow.add_node("context_retrieval_node", context_retrieval_node)
    workflow.add_node("query_classification_node", query_classifier_node)
    workflow.add_node("command_generation_node", command_generator_node)
    workflow.add_node("information_generation_node", information_generator_node)
    workflow.add_node("tool_execution_node", tool_execution_node)
    workflow.add_node("prepare_final_result_node", prepare_final_result_node)
    workflow.add_node("display_result_node", display_result_node)

    # Set the entry point to the conversation context node
    workflow.set_entry_point("conversation_context_node")

    # Add standard edges
    workflow.add_edge("conversation_context_node", "domain_analysis_node")
    workflow.add_edge("prepare_final_result_node", "display_result_node")
    workflow.add_edge("display_result_node", END)

    # Add conditional edges for domain analysis and context retrieval
    workflow.add_conditional_edges(
        "domain_analysis_node",
        check_domains_to_process,
        {
            "context_retrieval_node": "context_retrieval_node",
            "query_classification_node": "query_classification_node",
        },
    )

    workflow.add_conditional_edges(
        "context_retrieval_node",
        check_domains_to_process,
        {
            "context_retrieval_node": "context_retrieval_node",
            "query_classification_node": "query_classification_node",
        },
    )

    # Add conditional edge for query type branching
    workflow.add_conditional_edges(
        "query_classification_node",
        branch_on_query_type,
        {
            "command_generation_node": "command_generation_node",
            "information_generation_node": "information_generation_node",
        },
    )

    # Add conditional edges for tool usage
    workflow.add_conditional_edges(
        "information_generation_node",
        check_for_tool_usage,
        {
            "tool_execution_node": "tool_execution_node",
            "prepare_final_result_node": "prepare_final_result_node",
        },
    )

    workflow.add_conditional_edges(
        "command_generation_node",
        check_for_tool_usage,
        {
            "tool_execution_node": "tool_execution_node",
            "prepare_final_result_node": "prepare_final_result_node",
        },
    )

    # Add conditional edges for post-tool routing
    workflow.add_conditional_edges(
        "tool_execution_node",
        route_after_tool,
        {
            "command_generation_node": "command_generation_node",
            "information_generation_node": "information_generation_node",
            "prepare_final_result_node": "prepare_final_result_node",
        },
    )

    # Compile the graph
    return workflow.compile(checkpointer=MemorySaver())
