from os_assistant.core.state import AssistantState
from os_assistant.pydantic_models.schemas import (
    ContextRetrievalDetails,
    FinalResult,
    InformationResponse,
    VisionAnalysisDetails,
)
from os_assistant.utils import LOGGER


def add_mode_note_to_response(response, is_code_execution_enabled):
    """Add a mode-specific note to a response if needed"""
    if not is_code_execution_enabled and "disabled in the current mode" not in response:
        return (
            response
            + "\n\nNote: This response was generated without using the code execution tool, which is disabled in the current mode. It is based on general knowledge."
        )
    return response


def prepare_final_result_node(state: AssistantState) -> AssistantState:
    """Prepare the final result"""
    LOGGER.info("\nNODE: prepare_final_result_node")

    # Ensure domain_analysis and query_type exist before accessing keys
    domains_tmp = state.get("domain_analysis")
    if domains_tmp is None:
        LOGGER.warning("Domain analysis missing, using available domains for final result.")
        domains = state.get("domains", [])  # Fallback to available domains
    else:
        domains = getattr(domains_tmp, "domains", state.get("domains", []))

    query_type_tmp = state.get("query_type")
    if query_type_tmp is None:
        LOGGER.warning(
            "Query type missing, defaulting to 'information' for final result."
        )
        response_type = "information"  # Fallback type
    else:
        # Support both dict-like and object with attribute
        response_type = (
            query_type_tmp.get("query_type")
            if isinstance(query_type_tmp, dict)
            else getattr(query_type_tmp, "query_type", "information")
        )

    # Create context summary
    context_summary = "Analyzed information from: "
    context_summary += ", ".join(domains)

    # Determine response content
    match response_type:
        case "command":
            if state.get("command_response"):
                response = state["command_response"]
            else:
                LOGGER.warning("Command response expected but missing.")
                # Create a fallback command response if needed, or switch type
                response_type = "information"  # Switch to info if command failed
                response = InformationResponse(
                    answer=f"Could not generate a command for '{state['prompt']}'. Please try rephrasing.",
                    sources=["System processing error"],
                )
        # Handle information response (either primary or fallback)
        case "information":
            if state.get("information_response"):
                response = state["information_response"]
            else:
                LOGGER.warning("Information response expected but missing.")
                # Create a fallback information response
                response = InformationResponse(
                    answer=f"Unable to generate an answer for '{state['prompt']}' based on the available information.",
                    sources=["System processing error"],
                )

    # Ensure response is not None before creating FinalResult
    if response is None:
        LOGGER.error("Could not determine a valid response for the final result.")
        # Handle this case, maybe set final_result to an error state or raise exception
        # For now, create a minimal error response
        response = InformationResponse(
            answer="An unexpected error occurred while generating the response.",
            sources=["System error"],
        )
        response_type = "information"  # Ensure type matches the fallback

    # Build context retrieval details from state
    context_retrieval = _build_context_retrieval_details(state, domains)

    # Build vision analysis details from state
    vision_analysis = _build_vision_analysis_details(state)

    # Create final result
    import os

    final_result = FinalResult(
        query=state.get("original_prompt", state.get("prompt", "")),
        domains=domains,
        response_type=response_type,  # Use the potentially updated response_type
        response=response,  # Pass the dictionary directly
        context_summary=context_summary,
        context_retrieval=context_retrieval,
        vision_analysis=vision_analysis,
        prompt_variant=os.getenv("PROMPT_VARIANT"),
    )

    state["final_result"] = final_result

    return state


def _build_context_retrieval_details(
    state: AssistantState, domains: list[str]
) -> ContextRetrievalDetails | None:
    """
    Build ContextRetrievalDetails from the assistant state.

    Args:
        state: The assistant state containing retrieval information
        domains: List of domains that were processed

    Returns:
        ContextRetrievalDetails or None if no retrieval was performed
    """
    # Check if any retrieval was performed
    retrieval_sources = state.get("retrieval_sources", [])
    query_intent = state.get("query_intent")
    sql_context = state.get("sql_context")
    contexts = state.get("contexts", {})

    # If no retrieval sources, return None
    if not retrieval_sources and not query_intent and not sql_context and not contexts:
        return None

    # Extract RAG context from contexts dict
    rag_context = None
    rag_doc_count = 0
    combined_context = None

    if contexts:
        # Combine all domain contexts
        context_parts = []
        for domain, ctx in contexts.items():
            if ctx:
                context_parts.append(ctx)
                # Count RAG documents (look for "Log #" pattern)
                rag_doc_count += ctx.count("Log #")

        if context_parts:
            combined_context = "\n\n".join(context_parts)
            # If we have RAG in sources, the combined context includes RAG
            if "RAG Semantic Search" in retrieval_sources or "RAG" in retrieval_sources:
                rag_context = combined_context

    # Count SQL rows if we have SQL context
    sql_row_count = 0
    if sql_context:
        # Try to extract row count from formatted SQL context
        import re

        match = re.search(r"\((\d+) rows?\)", sql_context)
        if match:
            sql_row_count = int(match.group(1))

    return ContextRetrievalDetails(
        query_intent=query_intent,
        retrieval_sources=retrieval_sources if retrieval_sources else [],
        sql_context=sql_context,
        sql_query=None,  # We don't store the raw query in state currently
        sql_row_count=sql_row_count,
        rag_context=rag_context,
        rag_doc_count=rag_doc_count,
        combined_context=combined_context,
        domains_processed=domains if domains else [],
    )


def _build_vision_analysis_details(
    state: AssistantState,
) -> VisionAnalysisDetails | None:
    """
    Build VisionAnalysisDetails from the assistant state.

    Args:
        state: The assistant state containing vision analysis information

    Returns:
        VisionAnalysisDetails or None if no vision analysis was performed
    """
    vision_analysis = state.get("vision_analysis")

    # If no vision analysis in state, return None
    if not vision_analysis:
        return None

    # Handle error case
    if "error" in vision_analysis and not vision_analysis.get("success", False):
        return VisionAnalysisDetails(
            success=False,
            error=vision_analysis.get("error"),
        )

    # Build full vision analysis details
    return VisionAnalysisDetails(
        success=vision_analysis.get("success", True),
        extracted_text=vision_analysis.get("extracted_text"),
        error_codes=vision_analysis.get("error_codes", []),
        screenshot_type=vision_analysis.get("screenshot_type"),
        analysis=vision_analysis.get("analysis"),
        suggested_actions=vision_analysis.get("suggested_actions", []),
        error=vision_analysis.get("error"),
    )
