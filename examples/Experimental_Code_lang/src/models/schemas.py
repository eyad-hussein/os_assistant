from typing import List, Literal, Union, Optional
from pydantic import BaseModel, Field, field_validator


class DomainAnalysis(BaseModel):
    """Model for domain analysis results"""
    domains: List[str] = Field(..., description="List of relevant domains (e.g., ['filesystem', 'users'])")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score between 0 and 1")
    reasoning: str = Field(..., description="Explanation of why these domains were selected")

    @field_validator('domains')
    def check_domains_list(cls, v):
        if not isinstance(v, list):
            raise ValueError('domains must be a list')
        return v

class ContextResult(BaseModel):
    """Model for context retrieval results (if needed separately, currently context is stored directly in state)"""
    context: str = Field(..., description="Retrieved context")
    domain: str = Field(..., description="Domain of the context")

class QueryTypeResult(BaseModel):
    """Model for query type classification"""
    query_type: Literal["command", "information"] = Field(...,
                description="Type of query: 'command' or 'information'")
    reasoning: str = Field(..., description="Explanation for the classification")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in classification (0 to 1)")

class CommandResponse(BaseModel):
    """Model for command generation response"""
    command: str = Field(..., description="The generated Linux command string")
    explanation: str = Field(..., description="Explanation of what the command does")
    security_notes: Optional[str] = Field(default=None, description="Any security warnings or considerations")

class InformationResponse(BaseModel):
    """Model for information retrieval response"""
    answer: str = Field(..., description="The answer to the user's query based on context")
    sources: List[str] = Field(default_factory=list, description="List of sources or domains used for the answer")

class FinalResult(BaseModel):
    """Final result combining all outputs"""
    query: str = Field(..., description="Original user query")
    domains: List[str] = Field(..., description="List of domains identified as relevant")
    response_type: Literal["command", "information"] = Field(...,
                    description="The type of response generated ('command' or 'information')")
    # Use Union for the response field, matching the Pydantic models
    response: Union[CommandResponse, InformationResponse] = Field(...,
                    description="The actual response content (either CommandResponse or InformationResponse)")
    context_summary: str = Field(..., description="Summary of the context sources used (e.g., 'Analyzed information from: filesystem, users')")

