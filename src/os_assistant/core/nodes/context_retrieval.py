"""
Context Retrieval Node - ENHANCED with Hybrid RAG + SQL.

This node retrieves context using:
1. Structured SQL queries via MCP (for precise data)
2. Semantic RAG search (for understanding and summarization)
3. Hybrid approach (both) based on query analysis
"""

from tracer.config import LogDomain

from os_assistant.core.nodes.helpers import is_mcp_enabled, is_rag_enabled
from os_assistant.core.state import AssistantState
from os_assistant.tools.agentic_rag.application.search import search_logs
from os_assistant.utils import LOGGER
from os_assistant.utils.settings import MCP_FALLBACK_TO_RAG


def _get_or_create_context_details(state: AssistantState) -> dict:
    """
    Get or create context_retrieval_details dict, ensuring it persists in state.

    Args:
        state: Assistant state

    Returns:
        The context_retrieval_details dict (same reference as in state)
    """
    if state.get("context_retrieval_details") is None:
        state["context_retrieval_details"] = {
            "query_intent": None,
            "retrieval_sources": [],
            "sql_context": None,
            "sql_query": None,
            "sql_row_count": 0,
            "rag_context": None,
            "rag_doc_count": 0,
            "combined_context": None,
            "domains_processed": [],
        }
    return state["context_retrieval_details"]


def context_retrieval_node(state: AssistantState) -> AssistantState:
    """
    Retrieve context using hybrid RAG + SQL approach.

    This enhanced node:
    1. Routes queries to SQL, RAG, or both based on intent
    2. Executes SQL queries via MCP for structured data
    3. Executes RAG search for semantic understanding
    4. Fuses results into comprehensive context
    """
    LOGGER.info("\nNODE: context_retrieval_node (Hybrid RAG + SQL)")

    # Initialize retrieval tracking
    if "retrieval_sources" not in state or state["retrieval_sources"] is None:
        state["retrieval_sources"] = []

    # Initialize context retrieval details tracking - get reference
    details = _get_or_create_context_details(state)

    # Check if we have domains to process
    if not state.get("domains_to_process"):
        LOGGER.warning("No more domains to process for context retrieval.")
        return state

    current_domain = state["domains_to_process"].pop(0)
    state["current_domain"] = current_domain

    LOGGER.info(f"\nRetrieving context for domain: {current_domain}")

    # Check what retrieval methods are available
    mcp_available = is_mcp_enabled()
    rag_available = is_rag_enabled()

    if not mcp_available and not rag_available:
        LOGGER.warning("Both MCP and RAG are disabled. No context retrieval available.")
        state["domains_to_process"] = []
        state["current_domain"] = None
        return state

    try:
        # Route the query to determine intent
        if mcp_available:
            sql_context, rag_context = _execute_hybrid_retrieval(
                query=state["prompt"],
                domain=current_domain,
                state=state,
                rag_available=rag_available,
            )
        else:
            # MCP not available, fall back to RAG only
            sql_context = None
            rag_context = _execute_rag_search(state["prompt"], current_domain, state)

        # Fuse the results
        final_context = _fuse_contexts(sql_context, rag_context, state)

        # Store the context
        state["contexts"][current_domain] = final_context

        # Update context retrieval details - use the reference we got earlier
        details = _get_or_create_context_details(state)
        details["combined_context"] = final_context
        if current_domain not in details["domains_processed"]:
            details["domains_processed"].append(current_domain)

        # Force reassignment to ensure state updates (for TypedDict compatibility)
        state["context_retrieval_details"] = details

        LOGGER.info(f"Retrieved hybrid context from {current_domain}")
        LOGGER.info(
            f"Final context_retrieval_details: {state['context_retrieval_details']}"
        )

    except Exception as e:
        LOGGER.error(f"Error in hybrid retrieval for {current_domain}: {str(e)}")

        # Try fallback to RAG if configured
        if MCP_FALLBACK_TO_RAG and rag_available:
            LOGGER.info("Falling back to RAG-only retrieval...")
            try:
                rag_context = _execute_rag_search(
                    state["prompt"], current_domain, state
                )
                state["contexts"][current_domain] = rag_context or f"Error: {str(e)}"
                state["retrieval_sources"].append("RAG (fallback)")
            except Exception as rag_error:
                state["contexts"][current_domain] = (
                    f"Error retrieving context for {current_domain}: {str(e)} "
                    f"(RAG fallback also failed: {str(rag_error)})"
                )
        else:
            state["contexts"][current_domain] = (
                f"Error retrieving context for {current_domain}: {str(e)}"
            )

    # Clear current_domain after processing
    state["current_domain"] = None
    return state


def _execute_hybrid_retrieval(
    query: str,
    domain: str,
    state: AssistantState,
    rag_available: bool,
) -> tuple[str | None, str | None]:
    """
    Execute hybrid retrieval based on query routing.

    Args:
        query: User's query
        domain: Current domain being processed
        state: Assistant state
        rag_available: Whether RAG is available

    Returns:
        Tuple of (sql_context, rag_context)
    """
    from os_assistant.tools.mcp_client import (
        QueryIntent,
        get_mcp_client,
        get_query_router,
    )

    sql_context = None
    rag_context = None

    # Get reference to details dict
    details = _get_or_create_context_details(state)

    try:
        # Get MCP client and router
        mcp_client = get_mcp_client()

        # Test MCP connection first
        if not mcp_client.test_connection():
            LOGGER.warning("MCP server not available, falling back to RAG only")
            if rag_available:
                rag_context = _execute_rag_search(query, domain, state)
                state["retrieval_sources"].append("RAG")
            return sql_context, rag_context

        # Get schema for router
        schema = mcp_client.get_all_schemas()
        LOGGER.info(f"MCP schema retrieved with {len(schema)} tables for routing")
        LOGGER.info(f"MCP schema details: {schema}")
        router = get_query_router(schema)

        # Route the query
        decision = router.route(query)
        state["query_intent"] = decision.intent.value
        details["query_intent"] = decision.intent.value

        LOGGER.info(
            f"Query routed to: {decision.intent.value} "
            f"(confidence: {decision.confidence})"
        )

        # Store the SQL query from the router decision IMMEDIATELY
        if decision.sql_query:
            details["sql_query"] = decision.sql_query
            state["sql_query_executed"] = decision.sql_query
            LOGGER.info(f"Stored SQL query from router decision: {decision.sql_query}")

        # Execute based on intent
        if decision.intent == QueryIntent.STRUCTURED:
            # SQL only
            sql_context = _execute_sql_query(mcp_client, decision.sql_query, state)

        elif decision.intent == QueryIntent.SEMANTIC:
            # RAG only
            if rag_available:
                rag_context = _execute_rag_search(query, domain, state)
            else:
                LOGGER.warning("RAG not available for semantic query")

        elif decision.intent == QueryIntent.HYBRID:
            # Both SQL and RAG
            if decision.sql_query:
                sql_context = _execute_sql_query(mcp_client, decision.sql_query, state)

            if rag_available:
                search_query = decision.rag_query or query
                rag_context = _execute_rag_search(search_query, domain, state)

        # Force reassignment to ensure state updates
        state["context_retrieval_details"] = details

    except ImportError as e:
        LOGGER.error(f"MCP client import error: {e}")
        # Fall back to RAG
        if rag_available:
            rag_context = _execute_rag_search(query, domain, state)
            state["retrieval_sources"].append("RAG")

    except Exception as e:
        LOGGER.error(f"Hybrid retrieval error: {e}")
        # Try RAG as fallback
        if rag_available and MCP_FALLBACK_TO_RAG:
            rag_context = _execute_rag_search(query, domain, state)
            state["retrieval_sources"].append("RAG (fallback)")

    return sql_context, rag_context


def _execute_sql_query(mcp_client, sql_query: str, state: AssistantState) -> str | None:
    """
    Execute SQL query via MCP.

    Args:
        mcp_client: MCP client instance
        sql_query: SQL query string
        state: Assistant state

    Returns:
        Formatted SQL result string
    """
    from os_assistant.tools.mcp_client.result_fusion import get_result_fusion

    if not sql_query:
        return None

    LOGGER.info(f"Executing SQL: {sql_query}...")

    # Get reference to details dict
    details = _get_or_create_context_details(state)

    # Store the SQL query that was executed
    state["sql_query_executed"] = sql_query
    details["sql_query"] = sql_query
    LOGGER.info(f"Set sql_query in details: {details['sql_query']}")

    result = mcp_client.execute_sql_query(sql_query)
    fusion = get_result_fusion()

    if result.success:
        formatted, row_count = fusion._format_sql_result(result)
        state["sql_context"] = formatted
        details["sql_context"] = formatted
        details["sql_row_count"] = row_count
        state["retrieval_sources"].append("SQL Database")
        if "SQL Database" not in details["retrieval_sources"]:
            details["retrieval_sources"].append("SQL Database")

        # Force reassignment to ensure state updates
        state["context_retrieval_details"] = details

        LOGGER.info(f"SQL returned {row_count} rows")
        LOGGER.info(
            f"context_retrieval_details after SQL: {state['context_retrieval_details']}"
        )
        return formatted
    else:
        LOGGER.error(f"SQL query failed: {result.error}")
        return f"SQL Query Error: {result.error}"


def _execute_rag_search(query: str, domain: str, state: AssistantState) -> str | None:
    """
    Execute RAG semantic search (original implementation).

    Args:
        query: Search query
        domain: Domain to search in
        state: Assistant state

    Returns:
        Formatted RAG context string
    """
    # Get reference to details dict
    details = _get_or_create_context_details(state)

    try:
        # Convert domain string to LogDomain enum
        try:
            domain_enum = LogDomain(domain.strip())
        except (KeyError, ValueError):
            LOGGER.warning(
                f"Domain {domain} not found in LogDomain enum. Using FS as fallback."
            )
            domain_enum = LogDomain.FS

        # Call search_logs from Agentic_RAG
        logs, summaries = search_logs(
            query=query,
            domains=[domain_enum],
            top_k=3,
            summarize=True,
            auto_init=True,
        )

        # Format the results into context
        context = ""
        if logs:
            for i, log in enumerate(logs):
                domain_info = f"Domain: {log.get('domain', domain_enum.name)}\n"
                context += (
                    f"{domain_info}Log #{log['log_number']} "
                    f"(Timestamp: {log['timestamp']})\n"
                )

                # Include summary if available
                if summaries and i < len(summaries):
                    context += f"Summary: {summaries[i]}\n"

                # Add the log text
                context += f"Content: {log['log_text']}\n\n"

            state["retrieval_sources"].append("RAG Semantic Search")
            if "RAG Semantic Search" not in details["retrieval_sources"]:
                details["retrieval_sources"].append("RAG Semantic Search")
            details["rag_context"] = context
            details["rag_doc_count"] = len(logs)

            # Force reassignment to ensure state updates
            state["context_retrieval_details"] = details
        else:
            context = f"No relevant logs found for query: '{query}' in domain {domain}"

        LOGGER.info(f"RAG returned {len(logs) if logs else 0} documents")
        return context

    except Exception as e:
        LOGGER.error(f"RAG search error: {e}")
        return None


def _fuse_contexts(
    sql_context: str | None,
    rag_context: str | None,
    state: AssistantState,
) -> str:
    """
    Fuse SQL and RAG contexts into unified context.

    Args:
        sql_context: Context from SQL query
        rag_context: Context from RAG search
        state: Assistant state

    Returns:
        Combined context string
    """
    from os_assistant.tools.mcp_client.result_fusion import get_result_fusion

    # Get reference to details dict - preserve sql_query!
    details = _get_or_create_context_details(state)
    preserved_sql_query = details.get("sql_query")
    LOGGER.info(f"Before fusion - preserved sql_query: {preserved_sql_query}")

    fusion = get_result_fusion()
    fused = fusion.fuse(
        sql_result=sql_context,  # Already formatted
        rag_result=rag_context,
        query=state.get("prompt", ""),
        sql_query=preserved_sql_query,  # Pass through the sql_query
    )

    # Update context_retrieval_details with fused result data
    # BUT preserve sql_query which was already set
    if not details.get("sql_context") and fused.sql_context:
        details["sql_context"] = fused.sql_context
    if not details.get("sql_row_count") and fused.sql_row_count:
        details["sql_row_count"] = fused.sql_row_count
    if not details.get("rag_context") and fused.rag_context:
        details["rag_context"] = fused.rag_context
    if not details.get("rag_doc_count") and fused.rag_doc_count:
        details["rag_doc_count"] = fused.rag_doc_count

    # IMPORTANT: Restore sql_query if it was lost
    if preserved_sql_query and not details.get("sql_query"):
        details["sql_query"] = preserved_sql_query

    # Always update combined_context from fusion
    details["combined_context"] = fused.combined_context

    # Merge retrieval sources (avoid duplicates)
    existing_sources = set(details.get("retrieval_sources", []))
    for source in fused.sources:
        existing_sources.add(source)
    details["retrieval_sources"] = list(existing_sources)

    # Force reassignment to ensure state updates
    state["context_retrieval_details"] = details
    LOGGER.info(
        f"After fusion - context_retrieval_details: {state['context_retrieval_details']}"
    )

    return fused.combined_context
