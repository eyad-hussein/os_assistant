from typing import Any, TypedDict

from os_assistant.pydantic_models.schemas import (
    CommandResponse,
    DomainAnalysis,
    FinalResult,
    InformationResponse,
    QueryTypeResult,
)


class AssistantState(TypedDict, total=False):
    """State for the assistant LangGraph with improved type hints"""

    # Input
    prompt: str  # User's original prompt
    original_prompt: str | None  # Prompt before context enhancement

    # Domain handling
    domains: list[str]  # List of all available domains
    domain_analysis: DomainAnalysis | None  # Result of domain analysis
    domains_to_process: list[str]  # Domains identified as relevant by analysis
    current_domain: str | None  # Domain being processed in the loop

    # Context storage
    contexts: dict[str, str]  # Domain -> Retrieved context string

    # Query classification
    query_type: QueryTypeResult | None  # Result of query classification

    # Response generation
    command_response: CommandResponse | None  # Generated command response
    information_response: InformationResponse | None  # Generated info response

    # Final result
    final_result: FinalResult | None  # Compiled final result

    # Tool handling
    tool_context: str | None  # Context for tool usage
    tool_question: str | None  # Question for tool usage
    tool_originating_node: str | None  # To track which node requested tools
    tool_usage_count: int  # Counter for tool usage
    raw_tool_results: str | None  # Raw output from tool execution
    tool_code: str | None  # Code used in tool execution
    tool_analysis: str | None  # Analysis of tool execution results

    # Conversation history
    conversation_history: list[dict[str, Any]]  # List of past queries and responses
    conversation_summary: str | None  # Summary of past interactions

    # Configuration
    assistant_mode: int  # Current assistant mode (0-3)
