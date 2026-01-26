from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

import os_assistant.core.nodes as nodes
from os_assistant.core.nodes.registry import (
    COMMAND_NODE,
    CONTEXT_NODE,
    CONV_CONTEXT_NODE,
    DISPLAY_NODE,
    DOMAIN_ANALYSIS_NODE,
    FINAL_NODE,
    INFO_NODE,
    QUERY_CLASS_NODE,
    TOOL_NODE,
    VISION_NODE,
)
from os_assistant.core.routing.logic import (
    branch_on_query_type,
    check_domains_to_process,
    check_for_tool_usage,
    route_after_tool,
)
from os_assistant.core.routing.rules import ROUTING_RULES
from os_assistant.core.state import AssistantState
from os_assistant.utils.settings import VISION_ENABLED

ROUTING_FUNCS = {
    "branch_on_query_type": branch_on_query_type,
    "check_for_tool_usage": check_for_tool_usage,
    "route_after_tool": route_after_tool,
    "check_domains_to_process": check_domains_to_process,
}


# --- Build the Graph ---
def build_assistant_graph():
    workflow = StateGraph(AssistantState)

    # Add nodes
    workflow.add_node(CONV_CONTEXT_NODE, nodes.conversation_context_node)
    workflow.add_node(DOMAIN_ANALYSIS_NODE, nodes.domain_analysis_node)
    workflow.add_node(CONTEXT_NODE, nodes.context_retrieval_node)
    workflow.add_node(QUERY_CLASS_NODE, nodes.query_classifier_node)
    workflow.add_node(COMMAND_NODE, nodes.command_generator_node)
    workflow.add_node(INFO_NODE, nodes.information_generator_node)
    workflow.add_node(TOOL_NODE, nodes.tool_execution_node)
    workflow.add_node(FINAL_NODE, nodes.prepare_final_result_node)
    workflow.add_node(DISPLAY_NODE, nodes.display_result_node)

    # Add vision node if enabled (processes attached images)
    if VISION_ENABLED:
        workflow.add_node(VISION_NODE, nodes.vision_analysis_node)

    # Entry point
    workflow.set_entry_point(CONV_CONTEXT_NODE)

    # Straight edges - with optional vision node
    if VISION_ENABLED:
        # Flow: CONV_CONTEXT -> VISION -> DOMAIN_ANALYSIS
        workflow.add_edge(CONV_CONTEXT_NODE, VISION_NODE)
        workflow.add_edge(VISION_NODE, DOMAIN_ANALYSIS_NODE)
    else:
        # Original flow: CONV_CONTEXT -> DOMAIN_ANALYSIS
        workflow.add_edge(CONV_CONTEXT_NODE, DOMAIN_ANALYSIS_NODE)

    workflow.add_edge(FINAL_NODE, DISPLAY_NODE)
    workflow.add_edge(DISPLAY_NODE, END)

    # Dynamic edges from routing rules
    for node_name, config in ROUTING_RULES.items():
        workflow.add_conditional_edges(
            node_name,
            ROUTING_FUNCS[config["func"]],
            config["routes"],
        )

    # Compile the graph
    return workflow.compile(checkpointer=MemorySaver())
