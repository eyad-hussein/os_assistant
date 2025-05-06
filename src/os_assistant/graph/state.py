from typing import Any, TypedDict

from os_assistant.models.schemas import (
    CommandResponse,
    DomainAnalysis,
    FinalResult,
    InformationResponse,
    QueryTypeResult,
)


class LinuxAssistantState(TypedDict):
    """State for the Linux assistant LangGraph"""

    # Input
    prompt: str
    domains: list[str]  # List of all available domains

    # Domain analysis
    domain_analysis: DomainAnalysis | None  # Result of domain analysis

    # Context retrieval
    contexts: dict[str, str]  # Domain -> Retrieved context string
    domains_to_process: list[str]  # Domains identified as relevant by analysis
    current_domain: str | None  # Domain being processed in the loop

    # Query classification
    query_type: QueryTypeResult  # Result of query classification

    # Response generation
    command_response: CommandResponse | None  # Generated command response
    information_response: InformationResponse | None  # Generated info response

    # Final result
    final_result: FinalResult | None  # Compiled final result

    # Add conversation history field to track past interactions
    conversation_history: list[dict[str, Any]]  # List of past queries and responses

    # Add summary field for potential conversation summarization
    conversation_summary: str | None  # Summary of past interactions
