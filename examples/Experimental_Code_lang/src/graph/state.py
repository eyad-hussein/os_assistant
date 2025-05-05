from typing import Any, Dict, List, Optional, TypedDict

from ..models.schemas import (
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
    domains: List[str]  # List of all available domains

    # Domain analysis
    domain_analysis: Optional[DomainAnalysis]  # Result of domain analysis

    # Context retrieval
    contexts: Dict[str, str]  # Domain -> Retrieved context string
    domains_to_process: List[str]  # Domains identified as relevant by analysis
    current_domain: Optional[str]  # Domain being processed in the loop

    # Query classification
    query_type: Optional[QueryTypeResult]  # Result of query classification

    # Response generation
    command_response: Optional[CommandResponse]  # Generated command response
    information_response: Optional[InformationResponse]  # Generated info response

    # Final result
    final_result: Optional[FinalResult]  # Compiled final result

    # Add conversation history field to track past interactions
    conversation_history: List[Dict[str, Any]]  # List of past queries and responses

    # Add summary field for potential conversation summarization
    conversation_summary: Optional[str]  # Summary of past interactions
