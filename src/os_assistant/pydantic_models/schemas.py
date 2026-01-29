from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ToolExecutionDetails(BaseModel):
    """Model for structured tool execution details"""

    question: str | None = Field(
        default=None, description="The question that was asked to the tool"
    )
    code: str | None = Field(
        default=None, description="The code that was executed by the tool"
    )
    raw_output: str | None = Field(
        default=None, description="Raw output from the tool execution"
    )
    analysis: str | None = Field(
        default=None, description="Analysis/interpretation of the tool results"
    )
    success: bool = Field(
        default=True, description="Whether the tool execution was successful"
    )
    error_message: str | None = Field(
        default=None, description="Error message if execution failed"
    )


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
    requires_logs: bool = Field(
        ..., description="Whether the query requires historical logs from the domains"
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


class ContextRetrievalDetails(BaseModel):
    """Model for detailed context retrieval information from hybrid RAG + SQL"""

    query_intent: str | None = Field(
        default=None,
        description="Query routing intent: 'structured' (SQL), 'semantic' (RAG), or 'hybrid' (both)",
    )
    retrieval_sources: list[str] = Field(
        default_factory=list,
        description="List of sources used for retrieval (e.g., 'SQL Database', 'RAG Semantic Search')",
    )
    sql_context: str | None = Field(
        default=None, description="Context retrieved from SQL database queries via MCP"
    )
    sql_query: str | None = Field(
        default=None, description="The SQL query that was executed"
    )
    sql_row_count: int = Field(
        default=0, description="Number of rows returned from SQL query"
    )
    rag_context: str | None = Field(
        default=None, description="Context retrieved from RAG semantic search"
    )
    rag_doc_count: int = Field(
        default=0, description="Number of documents retrieved from RAG"
    )
    combined_context: str | None = Field(
        default=None, description="Fused context from SQL and RAG"
    )
    domains_processed: list[str] = Field(
        default_factory=list, description="Domains that were processed for context"
    )


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

    command: str = Field(
        ...,
        description="The generated system-specific command (PowerShell, Linux, or Mac)",
    )
    security_notes: str | None = Field(
        default=None, description="Any security warnings or considerations"
    )
    what_command_does: str = Field(
        ..., description="Explanation of what the command does and how it works"
    )
    tool_execution: ToolExecutionDetails | None = Field(
        default=None, description="Structured details about tool execution if used"
    )
    # Legacy fields for backward compatibility
    tool_breakdown: str | None = Field(
        default=None, description="Breakdown of any tools used to generate the command"
    )
    tool_results: str | None = Field(
        default=None, description="Raw unedited results from tool execution"
    )
    tool_interpretation: str | None = Field(
        default=None, description="Interpretation of the tool results"
    )
    is_python_script: bool = Field(
        default=False,
        description="Whether the command is a Python script that should be saved and executed",
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
    tool_execution: ToolExecutionDetails | None = Field(
        default=None, description="Structured details about tool execution if used"
    )
    # Legacy fields for backward compatibility
    tool_breakdown: str | None = Field(
        default=None, description="Breakdown of any tools used to gather information"
    )
    tool_results: str | None = Field(
        default=None, description="Raw unedited results from tool execution"
    )
    tool_interpretation: str | None = Field(
        default=None, description="Interpretation of the tool results"
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
    context_retrieval: ContextRetrievalDetails | None = Field(
        default=None,
        description="Detailed information about how context was retrieved (SQL, RAG, hybrid)",
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


class CodeExecuteRequest(BaseModel):
    """Model for code execution tool requests"""

    question: str = Field(
        ..., description="The question to be answered using code execution"
    )
    name: str = Field(
        default="code_execute_tool", description="The name of the tool to call"
    )
