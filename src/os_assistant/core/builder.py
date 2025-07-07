from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from os_assistant.core.routing_rules import ROUTING_RULES
from os_assistant.core.routing_logic import *
from os_assistant.core.nodes_registry import *
from os_assistant.utils.settings import ASSISTANT_MODE
from os_assistant.core import nodes
from os_assistant.core.state import AssistantState

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

    # Entry point
    workflow.set_entry_point(CONV_CONTEXT_NODE)

    # Straight edges
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
