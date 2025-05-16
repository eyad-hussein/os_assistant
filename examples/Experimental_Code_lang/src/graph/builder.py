from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from .edges import branch_on_query_type, check_domains_to_process
from .nodes import (
    command_generator_node,
    context_retrieval_node,
    conversation_context_node,  # Add the new node import
    display_result_node,
    domain_analysis_node,
    information_generator_node,
    prepare_final_result_node,
    query_classifier_node,
)
from .state import LinuxAssistantState

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

    # Add standard edges for the rest of the flow
    workflow.add_edge("command_generation_node", "prepare_final_result_node")
    workflow.add_edge("information_generation_node", "prepare_final_result_node")
    workflow.add_edge("prepare_final_result_node", "display_result_node")
    workflow.add_edge("display_result_node", END)

    # Compile the graph
    return workflow.compile(checkpointer=MemorySaver())
