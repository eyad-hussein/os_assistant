from os_assistant.core.nodes.registry import *

ROUTING_RULES = {
    DOMAIN_ANALYSIS_NODE: {
        "func": "check_domains_to_process",
        "routes": {
            CONTEXT_NODE: CONTEXT_NODE,
            QUERY_CLASS_NODE: QUERY_CLASS_NODE,
        },
    },
    CONTEXT_NODE: {
        "func": "check_domains_to_process",
        "routes": {
            CONTEXT_NODE: CONTEXT_NODE,
            QUERY_CLASS_NODE: QUERY_CLASS_NODE,
        },
    },
    QUERY_CLASS_NODE: {
        "func": "branch_on_query_type",
        "routes": {
            COMMAND_NODE: COMMAND_NODE,
            INFO_NODE: INFO_NODE,
        },
    },
    COMMAND_NODE: {
        "func": "check_for_tool_usage",
        "routes": {
            TOOL_NODE: TOOL_NODE,
            FINAL_NODE: FINAL_NODE,
        },
    },
    INFO_NODE: {
        "func": "check_for_tool_usage",
        "routes": {
            TOOL_NODE: TOOL_NODE,
            FINAL_NODE: FINAL_NODE,
        },
    },
    TOOL_NODE: {
        "func": "route_after_tool",
        "routes": {
            COMMAND_NODE: COMMAND_NODE,
            INFO_NODE: INFO_NODE,
            FINAL_NODE: FINAL_NODE,
        },
    },
}
