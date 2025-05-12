from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class CodeAnalysis(BaseModel):
    """Model for parsed code analysis response"""
    code: str = Field(description="The final executable Python code using subprocess or os to achieve the instruction.")
    dangerous: int = Field(ge=1, le=3, description="Danger level: 1 (safe), 2 (caution), 3 (dangerous)")
    reason: str = Field(description="Explanation of why the code is dangerous or safe")
    
class CodeExecutionState(BaseModel):
    """State management for code execution agent"""
    
    question: str = Field(
        default="",
        description="The input question or code snippet to be executed."
    )
    
    code: str = Field(
        default="",
        description="The code to be executed or that was generated."
    )
    
    error_code: Optional[str] = Field(
        default=None,
        description="Any error messages or exceptions encountered during code execution."
    )
    
    execution_result: Optional[str] = Field(
        default=None,
        description="The Result of the code execution"
    )
    
    agent_output: Optional[Any] = Field(
        default=None,
        description="The Final output what the model did after finishing."
    )
    
    danger_analysis: Optional[Dict] = Field(
        default=None,
        description="Analysis of code's danger level and reason"
    )
