from .state import LinuxAssistantState


def check_domains_to_process(state: LinuxAssistantState) -> str:
    """Check if there are more domains to process for context retrieval"""
    if state.get("domains_to_process"): # Check if list exists and is not empty
        print(f"Next domain for context: {state['domains_to_process'][0]}")
        return "context_retrieval_node"
    else:
        print("All relevant domains processed for context. Moving to query classification.")
        return "query_classification_node"

def branch_on_query_type(state: LinuxAssistantState) -> str:
    """Branch based on query type"""
    # Add safety check for query_type existence
    if state.get("query_type") and state["query_type"].get("query_type") == "command":
        print("Query classified as command. Moving to command generation.")
        return "command_generation_node"
    else:
        # Default to information if query_type is missing or not 'command'
        if not state.get("query_type"):
            print("Query type missing, defaulting to information generation.")
        else:
            print("Query classified as information. Moving to information generation.")
        return "information_generation_node"
