from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from os_assistant.graph.nodes import (
    command_generator_node,
    context_retrieval_node,
    conversation_context_node,  # Add the new node import
    display_result_node,
    domain_analysis_node,
    information_generator_node,
    prepare_final_result_node,
    query_classifier_node,
    tool_execution_node,
)
from os_assistant.graph.state import LinuxAssistantState

# --- Edges ----


def check_for_tool_usage(state: LinuxAssistantState) -> str:
    """Check if we need to route to tool execution"""
    if state.get("tool_originating_node") is not None:
        print("Tool usage detected. Routing to tool execution.")
        return "tool_execution_node"
    return "prepare_final_result_node"


def route_after_tool(state: LinuxAssistantState) -> str:
    """Route back to originating node after tool execution"""
    originating_node = state.get("tool_originating_node")

    # Clear the routing field
    state["tool_originating_node"] = None

    # Return to appropriate node
    if originating_node:
        return originating_node

    # Default fallback path
    return "prepare_final_result_node"


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


# --- Build the Graph ---


def build_linux_assistant_graph():
    """Build the LangGraph for the Linux assistant"""
    # Create a new graph
    workflow = StateGraph(LinuxAssistantState)

    # Add nodes
    workflow.add_node(
        "conversation_context_node", conversation_context_node
    )  # Add the new node
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

    # Add edge from conversation context to domain analysis
    workflow.add_edge("conversation_context_node", "domain_analysis_node")

    # Add conditional edges for context retrieval loop
    workflow.add_conditional_edges(
        "domain_analysis_node",
        check_domains_to_process,
        {
            "context_retrieval_node": "context_retrieval_node",
            "query_classification_node": "query_classification_node",  # Go directly if no domains found
        },
    )
    workflow.add_conditional_edges(
        "context_retrieval_node",
        check_domains_to_process,  # Check again after retrieving context for one domain
        {
            "context_retrieval_node": "context_retrieval_node",  # Loop back if more domains
            "query_classification_node": "query_classification_node",  # Continue if done
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
    # Add conditional edge after information generation
    workflow.add_conditional_edges(
        "information_generation_node",
        check_for_tool_usage,
        {
            "tool_execution_node": "tool_execution_node",
            "prepare_final_result_node": "prepare_final_result_node",
        },
    )
    workflow.add_conditional_edges(
        "tool_execution_node",
        route_after_tool,
        {
            "information_generation_node": "information_generation_node",
        },
    )

    # Add standard edges for the rest of the flow
    workflow.add_edge("command_generation_node", "prepare_final_result_node")
    workflow.add_edge("prepare_final_result_node", "display_result_node")
    workflow.add_edge("display_result_node", END)

    # Compile the graph
    return workflow.compile(checkpointer=MemorySaver())
