from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field


class MetricScore(BaseModel):
    """Score for a single evaluation metric."""

    score: float = Field(..., ge=1, le=5, description="Score on a 1-5 scale")
    explanation: str = Field(..., description="Explanation for the score")


class EvaluationScores(BaseModel):
    """Comprehensive evaluation scores across multiple dimensions."""

    correctness: float = Field(..., ge=1, le=5, description="Accuracy of the response")
    correctness_explanation: str = Field(
        ..., description="Explanation for correctness score"
    )

    completeness: float = Field(
        ..., ge=1, le=5, description="Coverage of all required information"
    )
    completeness_explanation: str = Field(
        ..., description="Explanation for completeness score"
    )

    clarity: float = Field(..., ge=1, le=5, description="Clarity and understandability")
    clarity_explanation: str = Field(..., description="Explanation for clarity score")


class EvaluationResult(BaseModel):
    """Result of evaluating an OS Assistant response."""

    scores: EvaluationScores
    overall_score: float = Field(..., ge=1, le=5)
    reasoning: str = Field(..., description="Overall reasoning for the evaluation")


class LatencyMetrics(BaseModel):
    """Detailed latency measurements for evaluation."""

    prompt_processing_ms: float
    correctness_evaluation_ms: float = 0.0
    completeness_evaluation_ms: float = 0.0
    clarity_evaluation_ms: float = 0.0
    total_evaluation_ms: float


class EvaluationSummary(BaseModel):
    """Summary of evaluation results."""

    total_samples: int
    average_score: float
    average_correctness: float
    average_completeness: float
    average_clarity: float
    command_average: Optional[float] = None
    information_average: Optional[float] = None
    domain_scores: Dict[str, float]
    latency_metrics: Dict[str, float]
    detailed_results: List[Dict]


class BatchSummary(BaseModel):
    """Summary of a batch of evaluation results."""

    batch_size: int
    average_score: float
    average_correctness: float
    average_completeness: float
    average_clarity: float
    timestamp: str


class DatasetSample(BaseModel):
    """A sample from the evaluation dataset."""

    question: str
    type: str
    expected_response: str
    domain: str
    agent_output: Optional[str] = None  # Add field for the actual agent output
    source_logs: Optional[List[int]] = None
    timestamps: Optional[List[str]] = None
    rag_enhanced: Optional[bool] = None
    rag_logs: Optional[List[int]] = None
    generated_type: Optional[str] = None


class EvaluationDataset(BaseModel):
    """Evaluation dataset structure."""

    metadata: Dict
    samples: List[DatasetSample]

    def to_serializable(self) -> Dict:
        """Convert the model to a serializable dictionary."""
        data = self.model_dump()
        # Process any fields that might not be serializable
        return data


class RunningMetrics(BaseModel):
    """Running metrics for an ongoing evaluation."""

    total_evaluated: int
    total_samples: int
    average_score: float
    average_correctness: float
    average_completeness: float
    average_clarity: float
    timestamp: str
    latency_metrics: Optional[Dict[str, float]] = None

    def to_serializable(self) -> Dict:
        """Convert the model to a serializable dictionary."""
        data = self.model_dump()
        # Process any fields that might not be serializable
        return data
