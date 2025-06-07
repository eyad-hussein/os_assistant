from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class SimplifiedScores(BaseModel):
    """Simplified evaluation scores focusing on correctness and completeness."""

    correctness: int = Field(..., ge=1, le=5, description="Accuracy of the response")
    completeness: int = Field(
        ..., ge=1, le=5, description="Coverage of all required information"
    )


class EvaluationResult(BaseModel):
    """Result of evaluating an OS Assistant response."""

    scores: SimplifiedScores
    overall_score: float = Field(..., ge=1, le=5)
    correctness_explanation: str = Field(
        default="", description="Explanation for the correctness score"
    )
    completeness_explanation: str = Field(
        default="", description="Explanation for the completeness score"
    )
    reasoning: str


class LatencyMetrics(BaseModel):
    """Detailed latency measurements for evaluation."""

    prompt_processing_ms: float
    llm_evaluation_ms: float
    total_evaluation_ms: float


class EvaluationSummary(BaseModel):
    """Summary of evaluation results."""

    total_samples: int
    average_score: float
    average_correctness: float
    average_completeness: float
    domain_scores: Dict[str, float]
    latency_metrics: Dict[str, float]  # Avg latencies
    detailed_results: List[Dict]


class BatchSummary(BaseModel):
    """Summary of a batch of evaluation results."""

    batch_size: int
    average_score: float
    average_correctness: float
    average_completeness: float
    timestamp: str


class DatasetSample(BaseModel):
    """A sample from the evaluation dataset."""

    question: str
    type: str
    expected_response: str
    domain: str
    source_logs: List[int] | None = None
    timestamps: List[str] | None = None
    rag_enhanced: bool | None = None
    rag_logs: List[int] | None = None
    generated_type: str | None = None


class EvaluationDataset(BaseModel):
    """Evaluation dataset structure."""

    metadata: Dict
    samples: List[DatasetSample]


class RunningMetrics(BaseModel):
    """Running metrics for an ongoing evaluation."""

    total_evaluated: int
    total_samples: int
    average_score: float
    average_correctness: float
    average_completeness: float
    timestamp: str
    latency_metrics: Optional[Dict[str, float]] = None
