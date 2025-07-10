from tracer.config import LogDomain
from os_assistant.core.state import AssistantState
from os_assistant.core.nodes.helpers import is_rag_enabled
from os_assistant.tools.agentic_rag.application.search import search_logs


def context_retrieval_node(state: AssistantState) -> AssistantState:
    """Retrieve context for a domain using Agentic_RAG search_logs"""
    print("\nNODE: context_retrieval_node")

    # Check if RAG is enabled
    if not is_rag_enabled():
        print("RAG disabled in current mode. Skipping context retrieval.")
        # Skip context retrieval by clearing domains to process
        state["domains_to_process"] = []
        return state

    if not state["domains_to_process"]:
        print("No more domains to process for context retrieval.")
        return state

    current_domain = state["domains_to_process"].pop(0)
    state["current_domain"] = current_domain

    print(f"\nRetrieving context for domain: {current_domain}")

    try:
        # Convert domain string to LogDomain enum
        try:
            domain_enum = LogDomain(current_domain.strip())
        except KeyError:
            print(
                f"Warning: Domain {current_domain} not found in LogDomain enum. Using FS as fallback."
            )
            domain_enum = LogDomain.FS

        # Call search_logs from Agentic_RAG
        logs, summaries = search_logs(
            query=state["prompt"],
            domains=[domain_enum],
            top_k=3,
            summarize=True,
            auto_init=True,
        )

        # Format the results into context for the state
        context = ""
        if logs:
            for i, log in enumerate(logs):
                domain_info = f"Domain: {log.get('domain', domain_enum.name)}\n"
                context += f"{domain_info}Log #{log['log_number']} (Timestamp: {log['timestamp']})\n"

                # Include summary if available
                if summaries and i < len(summaries):
                    context += f"Summary: {summaries[i]}\n"

                # Add the log text
                context += f"Content: {log['log_text']}\n\n"
        else:
            context = f"No relevant logs found for query: '{state['prompt']}' in domain {current_domain}"

        # Store the context
        state["contexts"][current_domain] = context
        print(f"Retrieved context from {current_domain} using Agentic_RAG")

    except Exception as e:
        print(f"Error retrieving context for {current_domain}: {str(e)}")
        state["contexts"][current_domain] = (
            f"Error retrieving context for {current_domain}: {str(e)}"
        )

    # Clear current_domain after processing
    state["current_domain"] = None
    return state