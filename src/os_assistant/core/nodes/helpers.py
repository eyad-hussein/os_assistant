from os_assistant.tools.code_agent.wrapper import code_execute_tool
from os_assistant.utils.settings import ASSISTANT_MODE

tools = [code_execute_tool]


def is_rag_enabled() -> bool:
    return ASSISTANT_MODE in [0, 2]


def is_code_execution_enabled() -> bool:
    return ASSISTANT_MODE in [0, 1]


def get_mode_description(mode):
    """Return a description of the current assistant mode"""
    modes = {
        0: "Full mode (RAG + Code Tool)",
        1: "Code Tool only",
        2: "RAG only",
        3: "Basic mode (no RAG, no Code Tool)",
    }
    return modes.get(mode, "Unknown mode")


def build_combined_context(state):
    """Build combined context from retrieved contexts"""
    combined_context = ""

    # Skip if RAG is disabled
    if not is_rag_enabled():
        return "RAG context retrieval is disabled in the current mode."

    # Get domains to use
    relevant_domains = (
        state["domain_analysis"].domains
        if state.get("domain_analysis")
        else state.get("domains", [])
    )

    # Build context string
    for domain in relevant_domains:
        context = state.get("contexts", {}).get(domain, "No context retrieved.")
        combined_context += f"--- {domain.upper()} DOMAIN ---\n{context}\n\n"

    if not combined_context:
        combined_context = "No specific context was retrieved for the relevant domains."

    return combined_context


def build_tool_context_info(state, force_no_tool=False):
    """Build tool context info for prompts"""
    tool_context_info = ""
    code_tool_enabled = is_code_execution_enabled() and not force_no_tool
    tool_usage_count = state.get("tool_usage_count", 0)

    # Add previous tool execution results if available
    if state.get("tool_context"):
        tool_context_info = f"""
        IMPORTANT: I've already executed the tool for you! The results are below:
        
        {state["tool_context"]}
        
        Use this information to create an appropriate response.
        """
        if code_tool_enabled and not force_no_tool:
            tool_context_info += (
                "You can request additional information with the tool if needed."
            )

    # Add mode-specific information
    if not code_tool_enabled:
        tool_context_info += """
        IMPORTANT: Code execution tool is disabled in the current mode.
        Generate a response based on general knowledge without using the tool.
        """
    elif force_no_tool:
        tool_context_info += f"""
        CRITICAL INSTRUCTION: You have already used the tool {tool_usage_count} times.
        YOU MUST NOW GENERATE A RESPONSE WITHOUT USING THE TOOL AGAIN.
        DO NOT REQUEST MORE INFORMATION - USE WHAT YOU HAVE.
        """

    return tool_context_info


def should_force_direct_response(state):
    """Determine if we should force a direct response without tool usage"""
    tool_usage_count = state.get("tool_usage_count", 0)
    return not is_code_execution_enabled() or tool_usage_count >= 3
