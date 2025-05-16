from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class DomainAnalysis(BaseModel):
    """Model for domain analysis results"""

    domains: list[str] = Field(
        ..., description="List of relevant domains (e.g., ['file_system', 'users'])"
    )
    confidence: float = Field(
        ..., ge=0, le=1, description="Confidence score between 0 and 1"
    )
    reasoning: str = Field(
        ..., description="Explanation of why these domains were selected"
    )

    @field_validator("domains")
    def check_domains_list(cls, v):
        if not isinstance(v, list):
            raise ValueError("domains must be a list")
        return v


class ContextResult(BaseModel):
    """Model for context retrieval results (if needed separately, currently context is stored directly in state)"""

    context: str = Field(..., description="Retrieved context")
    domain: str = Field(..., description="Domain of the context")


class QueryTypeResult(BaseModel):
    """Model for query type classification"""

    query_type: Literal["command", "information"] = Field(
        ..., description="Type of query: 'command' or 'information'"
    )
    reasoning: str = Field(..., description="Explanation for the classification")
    confidence: float = Field(
        ..., ge=0, le=1, description="Confidence in classification (0 to 1)"
    )


class CommandResponse(BaseModel):
    """Model for command generation response"""

    command: str = Field(..., description="The generated Linux command string")
    explanation: str = Field(..., description="Explanation of what the command does")
    security_notes: str | None = Field(
        default=None, description="Any security warnings or considerations"
    )


class InformationResponse(BaseModel):
    """Model for information retrieval response"""

    answer: str = Field(
        ..., description="The answer to the user's query based on context"
    )
    sources: list[str] = Field(
        default_factory=list,
        description="List of sources or domains used for the answer",
    )


class FinalResult(BaseModel):
    """Final result combining all outputs"""

    query: str = Field(..., description="Original user query")
    domains: list[str] = Field(
        ..., description="List of domains identified as relevant"
    )
    response_type: Literal["command", "information"] = Field(
        ..., description="The type of response generated ('command' or 'information')"
    )
    # Use Union for the response field, matching the Pydantic models
    response: CommandResponse | InformationResponse = Field(
        ...,
        description="The actual response content (either CommandResponse or InformationResponse)",
    )
    context_summary: str = Field(
        ...,
        description="Summary of the context sources used (e.g., 'Analyzed information from: file_system, networking')",
    )


class ConversationEntry(BaseModel):
    """Model for a single conversation entry in history"""

    timestamp: str = Field(
        ..., description="ISO format timestamp of when the interaction occurred"
    )
    query: str = Field(..., description="Original user query")
    refined_query: str | None = Field(
        None, description="Query after context enhancement (if applicable)"
    )
    domains: list[str] = Field(
        ..., description="Domains that were relevant to this query"
    )
    response_type: Literal["command", "information"] = Field(
        ..., description="Type of response provided"
    )
    response: dict[str, Any] | CommandResponse | InformationResponse = Field(
        ...,
        description="The response provided (either CommandResponse or InformationResponse)",
    )


class ConversationSummary(BaseModel):
    """Model for conversation summary"""

    summary: str = Field(..., description="Summary of the conversation context")
    key_topics: list[str] = Field(
        ..., description="Key topics discussed in the conversation"
    )
    last_updated: str = Field(
        ..., description="ISO format timestamp of when the summary was last updated"
    )
