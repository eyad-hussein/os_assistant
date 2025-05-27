import json
import re
from typing import Any, Dict

from .models import CommandScores, EvaluationResult, GeneralScores, InformationScores


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
                    "relevance": 0,
                    "clarity": 0,
                },
                "overall_score": 0.0,
                "reasoning": "Failed to parse LLM response: No JSON found",
            }

    except Exception as e:
        # Return error result
        return {
            "scores": {
                "correctness": 0,
                "completeness": 0,
                "relevance": 0,
                "clarity": 0,
            },
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

    # Determine the score type
    score_type = determine_score_type(scores)

    # Create the appropriate Pydantic model
    if score_type == "command":
        scores_model = CommandScores(**scores)
    elif score_type == "information":
        scores_model = InformationScores(**scores)
    else:
        scores_model = GeneralScores(**scores)

    # Create and validate the full evaluation result
    eval_result = EvaluationResult(
        scores=scores_model,
        overall_score=result.get("overall_score", 0.0),
        reasoning=result.get("reasoning", "No reasoning provided"),
    )

    # Return as dictionary
    return eval_result.model_dump()


def determine_score_type(scores: Dict[str, Any]) -> str:
    """Determine the type of scores based on the keys.

    Args:
        scores: The scores dictionary

    Returns:
        Score type: "command", "information", or "general"
    """
    command_keys = {
        "command_correctness",
        "command_efficiency",
        "safety_considerations",
        "explanation_quality",
    }
    info_keys = {"factual_accuracy", "completeness", "relevance", "clarity"}

    score_keys = set(scores.keys())

    if all(k in score_keys for k in command_keys):
        return "command"
    elif all(k in score_keys for k in info_keys):
        return "information"
    else:
        return "general"


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
