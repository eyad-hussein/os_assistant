from typing import Any, Dict

from pydantic import BaseModel, Field


class CodeAnalysis(BaseModel):
    """Model for parsed code analysis response"""

    code: str = Field(
        description="The final executable Python code using subprocess or os to achieve the instruction."
    )
    dangerous: int = Field(
        ge=1, le=3, description="Danger level: 1 (safe), 2 (caution), 3 (dangerous)"
    )
    reason: str = Field(description="Explanation of why the code is dangerous or safe")


class CodeExecutionState(BaseModel):
    """State management for code execution agent"""

    question: str = Field(
        default="", description="The input question or code snippet to be executed."
    )

    code: str = Field(
        default="", description="The code to be executed or that was generated."
    )

    error_code: str | None = Field(
        default=None,
        description="Any error messages or exceptions encountered during code execution.",
    )

    execution_result: str | None = Field(
        default=None, description="The Result of the code execution"
    )

    agent_output: Any | None = Field(
        default=None, description="The Final output what the model did after finishing."
    )

    danger_analysis: Dict | None = Field(
        default=None, description="Analysis of code's danger level and reason"
    )
