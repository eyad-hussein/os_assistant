import json
import re
from typing import Any, Dict

from .models import EvaluationResult, SimplifiedScores


def parse_evaluation_result(response: str) -> Dict[str, Any]:
    """Parse the LLM evaluation response into a structured dictionary.

    Args:
        response: The raw LLM response string

    Returns:
        Parsed evaluation result as a dictionary
    """
    try:
        # Try to extract JSON from the response
        json_match = re.search(r"({.*})", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
            # Parse the JSON
            result = json.loads(json_str)

            # Validate using Pydantic
            return validate_evaluation_result(result)

        else:
            # If no JSON found, create a basic error result
            return {
                "scores": {
                    "correctness": 0,
                    "completeness": 0,
                },
                "correctness_explanation": "Failed to parse LLM response: No JSON found",
                "completeness_explanation": "Failed to parse LLM response: No JSON found",
                "overall_score": 0.0,
                "reasoning": "Failed to parse LLM response: No JSON found",
            }

    except Exception as e:
        # Return error result
        return {
            "scores": {
                "correctness": 0,
                "completeness": 0,
            },
            "correctness_explanation": f"Failed to parse LLM response: {str(e)}",
            "completeness_explanation": f"Failed to parse LLM response: {str(e)}",
            "overall_score": 0.0,
            "reasoning": f"Failed to parse LLM response: {str(e)}",
        }


def validate_evaluation_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize the evaluation result using Pydantic.

    Args:
        result: The parsed result dictionary

    Returns:
        Validated evaluation result
    """
    # Get the scores dictionary
    scores = result.get("scores", {})

    # Extract correctness and completeness scores
    simplified_scores = {
        "correctness": scores.get("correctness", 0),
        "completeness": scores.get("completeness", 0),
    }

    # Create the simplified scores model
    scores_model = SimplifiedScores(**simplified_scores)

    # Extract explanations - they may be in the scores object or at the top level
    correctness_explanation = scores.get(
        "correctness_explanation",
        result.get("correctness_explanation", "No explanation provided"),
    )
    completeness_explanation = scores.get(
        "completeness_explanation",
        result.get("completeness_explanation", "No explanation provided"),
    )

    # Create and validate the full evaluation result
    eval_result = EvaluationResult(
        scores=scores_model,
        overall_score=result.get("overall_score", 0.0),
        correctness_explanation=correctness_explanation,
        completeness_explanation=completeness_explanation,
        reasoning=result.get("reasoning", "No reasoning provided"),
    )

    # Return as dictionary
    return eval_result.model_dump()


def extract_final_result(state: Dict[str, Any]) -> Dict[str, Any] | None:
    """Extract the final_result from the OS Assistant state.

    Args:
        state: The OS Assistant state dictionary

    Returns:
        The final result or None if not found
    """
    final_result = state.get("final_result")
    if not final_result:
        return None

    # Convert to dictionary if it's an object
    if hasattr(final_result, "__dict__"):
        final_result = final_result.__dict__

    # Extract the response based on type
    response_type = final_result.get("response_type")
    response = final_result.get("response")

    # Convert response to dictionary if needed
    if hasattr(response, "__dict__"):
        response = response.__dict__

    return {
        "query": final_result.get("query"),
        "domains": final_result.get("domains", []),
        "response_type": response_type,
        "response": response,
    }
