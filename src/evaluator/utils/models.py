from typing import Dict, List

from pydantic import BaseModel, Field


class CommandScores(BaseModel):
    """Scores for command evaluation."""

    command_correctness: int = Field(..., ge=1, le=5)
    command_efficiency: int = Field(..., ge=1, le=5)
    safety_considerations: int = Field(..., ge=1, le=5)
    explanation_quality: int = Field(..., ge=1, le=5)


class InformationScores(BaseModel):
    """Scores for information evaluation."""

    factual_accuracy: int = Field(..., ge=1, le=5)
    completeness: int = Field(..., ge=1, le=5)
    relevance: int = Field(..., ge=1, le=5)
    clarity: int = Field(..., ge=1, le=5)


class GeneralScores(BaseModel):
    """General evaluation scores."""

    correctness: int = Field(..., ge=1, le=5)
    completeness: int = Field(..., ge=1, le=5)
    relevance: int = Field(..., ge=1, le=5)
    clarity: int = Field(..., ge=1, le=5)


class EvaluationResult(BaseModel):
    """Result of evaluating an OS Assistant response."""

    scores: CommandScores | InformationScores | GeneralScores
    overall_score: float = Field(..., ge=1, le=5)
    reasoning: str


class EvaluationSummary(BaseModel):
    """Summary of evaluation results."""

    total_samples: int
    average_score: float
    command_average: float | None = None
    information_average: float | None = None
    domain_scores: Dict[str, float]
    detailed_results: List[Dict]


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
