import json
import re
from typing import Any, Dict, List, Tuple, Union

from .models import EvaluationResult, EvaluationScores


def parse_evaluation_result(response_text: str) -> Dict[str, Any]:
    """Parse the evaluation result from the LLM response.

    Args:
        response_text: The raw response text from the LLM

    Returns:
        Parsed evaluation result as a dictionary
    """
    try:
        # First attempt: try to parse the response as JSON directly
        return json.loads(response_text)
    except json.JSONDecodeError:
        # If direct parsing fails, try to extract structured data using regex
        try:
            return extract_structured_evaluation(response_text)
        except Exception as e:
            print(f"Error extracting structured evaluation: {str(e)}")

        # If all attempts fail, create a fallback response
        return {
            "scores": {
                "correctness": 3,
                "correctness_explanation": "Parsing failed",
                "completeness": 3,
                "completeness_explanation": "Parsing failed",
                "clarity": 3,
                "clarity_explanation": "Parsing failed",
            },
            "overall_score": 3.0,
            "reasoning": f"Failed to parse LLM response: {response_text[:100]}...",
        }


def extract_structured_evaluation(text: str) -> Dict[str, Any]:
    """Extract structured evaluation data from unstructured text with improved Windows focus.

    Args:
        text: The raw evaluation text

    Returns:
        Dictionary with structured evaluation data
    """
    # Extract scores using regex with expanded patterns
    scores = {}

    # Look for various score formats with enhanced patterns
    score_patterns = [
        # Standard patterns
        (r"(?:CORRECTNESS|CORRECTNESS SCORE):\s*(\d+(?:\.\d+)?)", "correctness"),
        (r"(?:COMPLETENESS|COMPLETENESS SCORE):\s*(\d+(?:\.\d+)?)", "completeness"),
        (r"(?:CLARITY|CLARITY SCORE):\s*(\d+(?:\.\d+)?)", "clarity"),
        # Windows-specific patterns
        (r"(?:WINDOWS CORRECTNESS|WINDOWS SCORE):\s*(\d+(?:\.\d+)?)", "correctness"),
        (
            r"(?:POWERSHELL CORRECTNESS|CMD CORRECTNESS):\s*(\d+(?:\.\d+)?)",
            "correctness",
        ),
        # Alternative formats
        (r"SCORE\s+(?:FOR|ON)\s+CORRECTNESS:?\s*(\d+(?:\.\d+)?)", "correctness"),
        (r"SCORE\s+(?:FOR|ON)\s+COMPLETENESS:?\s*(\d+(?:\.\d+)?)", "completeness"),
        (r"SCORE\s+(?:FOR|ON)\s+CLARITY:?\s*(\d+(?:\.\d+)?)", "clarity"),
        # Basic numeric patterns (last resort)
        (r"CORRECTNESS:?\s*(\d+(?:\.\d+)?)\s*/\s*5", "correctness"),
        (r"COMPLETENESS:?\s*(\d+(?:\.\d+)?)\s*/\s*5", "completeness"),
        (r"CLARITY:?\s*(\d+(?:\.\d+)?)\s*/\s*5", "clarity"),
    ]

    for pattern, name in score_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                score_value = float(match.group(1))
                # Ensure the score is within the valid range of 1-5
                if 1 <= score_value <= 5:
                    scores[name] = score_value
            except (ValueError, IndexError):
                continue  # Skip this match if conversion fails

    # Fill in missing scores with defaults
    for name in ["correctness", "completeness", "clarity"]:
        if name not in scores:
            scores[name] = 3.0  # Default if not found

    # Extract explanations with improved pattern matching
    explanation_patterns = [
        # Standard patterns
        (
            r"(?:CORRECTNESS EXPLANATION|JUSTIFICATION.*?CORRECTNESS):\s*(.*?)(?:\n\n|\n(?:[A-Z][A-Z\s]+:)|\Z)",
            "correctness_explanation",
        ),
        (
            r"(?:COMPLETENESS EXPLANATION|JUSTIFICATION.*?COMPLETENESS):\s*(.*?)(?:\n\n|\n(?:[A-Z][A-Z\s]+:)|\Z)",
            "completeness_explanation",
        ),
        (
            r"(?:CLARITY EXPLANATION|JUSTIFICATION.*?CLARITY):\s*(.*?)(?:\n\n|\n(?:[A-Z][A-Z\s]+:)|\Z)",
            "clarity_explanation",
        ),
        # Windows-specific patterns
        (
            r"(?:WINDOWS EXPLANATION|EXPLANATION FOR WINDOWS):\s*(.*?)(?:\n\n|\n(?:[A-Z][A-Z\s]+:)|\Z)",
            "correctness_explanation",
        ),
    ]

    for pattern, name in explanation_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            explanation_text = match.group(1).strip()
            # Only use if we got a substantial explanation (not just a few characters)
            if len(explanation_text) > 10:
                scores[name] = explanation_text

    # Ensure all explanation fields exist
    for base_name in ["correctness", "completeness", "clarity"]:
        explanation_key = f"{base_name}_explanation"
        if explanation_key not in scores:
            # Look for the explanation in the general text if specific pattern wasn't found
            sentences = re.split(r"(?<=[.!?])\s+", text)
            for sentence in sentences:
                if base_name.lower() in sentence.lower() and len(sentence) > 20:
                    scores[explanation_key] = sentence
                    break
            else:
                scores[explanation_key] = f"No {base_name} explanation found"

    # Extract overall reasoning with enhanced patterns
    reasoning_patterns = [
        r"(?:REASONING|OVERALL ASSESSMENT|JUSTIFICATION|OVERALL EVALUATION):\s*(.*?)(?:\n\n|\Z)",
        r"(?:SUMMARY|CONCLUSION):\s*(.*?)(?:\n\n|\Z)",
    ]

    reasoning = "No reasoning provided"
    for pattern in reasoning_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            extracted_reasoning = match.group(1).strip()
            if len(extracted_reasoning) > reasoning:
                reasoning = extracted_reasoning
                break

    # Calculate weighted overall score with more emphasis on correctness
    correctness_weight = 0.45
    completeness_weight = 0.35
    clarity_weight = 0.20

    weighted_score = (
        scores.get("correctness", 3.0) * correctness_weight
        + scores.get("completeness", 3.0) * completeness_weight
        + scores.get("clarity", 3.0) * clarity_weight
    )

    # Apply slight curve for mid-range scores to avoid harsh ratings
    if 2.5 <= weighted_score < 4.0:
        weighted_score += 0.2

    # Adjust score for Windows specificity
    windows_terms = ["windows", "powershell", "cmd", "command prompt", "registry"]
    if any(term in text.lower() for term in windows_terms):
        # This suggests the evaluation properly considered Windows context
        weighted_score = min(weighted_score + 0.1, 5.0)

    # Cap at 5.0
    overall_score = min(weighted_score, 5.0)

    return {
        "scores": scores,
        "overall_score": overall_score,
        "reasoning": reasoning,
    }


def calculate_overall_score(scores: Dict[str, float]) -> float:
    """Calculate the overall score with a more balanced approach.

    Weights correctness most heavily, followed by completeness, then clarity.
    Uses a slightly more generous curve to avoid overly harsh scoring.

    Args:
        scores: Dictionary of individual metric scores

    Returns:
        Overall weighted score
    """
    # Extract scores with defaults if missing
    correctness = scores.get("correctness", 3.0)
    completeness = scores.get("completeness", 3.0)
    clarity = scores.get("clarity", 3.0)

    # Weights: correctness (45%), completeness (35%), clarity (20%)
    # This puts more emphasis on functional correctness and completeness
    weighted_score = (correctness * 0.45) + (completeness * 0.35) + (clarity * 0.20)

    # Apply a slight curve to be more generous with mid-range scores
    # This helps avoid unnecessarily harsh ratings for responses that are mostly good
    if 2.5 <= weighted_score < 4.0:
        weighted_score += 0.2

    # Ensure we don't exceed 5.0
    return min(weighted_score, 5.0)


def validate_evaluation_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize the evaluation result using Pydantic.

    Args:
        result: The parsed result dictionary

    Returns:
        Validated evaluation result
    """
    try:
        # Extract scores from the dictionary
        scores = result.get("scores", {})

        # Extract correctness, completeness, and clarity with their explanations
        # Ensure scores are within valid range (1-5)
        correctness = scores.get("correctness", 3)
        correctness = min(max(float(correctness), 1.0), 5.0)  # Clamp between 1 and 5

        completeness = scores.get("completeness", 3)
        completeness = min(max(float(completeness), 1.0), 5.0)  # Clamp between 1 and 5

        clarity = scores.get("clarity", 3)
        clarity = min(max(float(clarity), 1.0), 5.0)  # Clamp between 1 and 5

        # Extract explanations with fallbacks
        correctness_explanation = scores.get(
            "correctness_explanation",
            result.get("correctness_explanation", "No explanation provided"),
        )
        completeness_explanation = scores.get(
            "completeness_explanation",
            result.get("completeness_explanation", "No explanation provided"),
        )
        clarity_explanation = scores.get(
            "clarity_explanation",
            result.get("clarity_explanation", "No explanation provided"),
        )

        # Create scores model
        scores_model = EvaluationScores(
            correctness=correctness,
            correctness_explanation=correctness_explanation,
            completeness=completeness,
            completeness_explanation=completeness_explanation,
            clarity=clarity,
            clarity_explanation=clarity_explanation,
        )

        # Create and validate the full evaluation result
        # Use the provided overall score or calculate our weighted version if not present
        overall_score = result.get("overall_score")
        if overall_score is None:
            overall_score = calculate_overall_score(
                {
                    "correctness": correctness,
                    "completeness": completeness,
                    "clarity": clarity,
                }
            )
        else:
            # Ensure overall score is within valid range
            overall_score = min(max(float(overall_score), 1.0), 5.0)

        eval_result = EvaluationResult(
            scores=scores_model,
            overall_score=overall_score,
            reasoning=result.get("reasoning", "No reasoning provided"),
        )

        # Return as dictionary
        return eval_result.model_dump()
    except Exception as e:
        print(f"Error validating evaluation result: {str(e)}")
        # Return a safe fallback result
        return {
            "scores": {
                "correctness": 3.0,
                "correctness_explanation": "Validation error occurred, using default score",
                "completeness": 3.0,
                "completeness_explanation": "Validation error occurred, using default score",
                "clarity": 3.0,
                "clarity_explanation": "Validation error occurred, using default score",
            },
            "overall_score": 3.0,
            "reasoning": f"Error validating evaluation result: {str(e)}. Using default scores.",
        }


def extract_final_result(state: Dict[str, Any]) -> Dict[str, Any] | Any:
    """Extract the final_result from the OS Assistant state.

    Args:
        state: The OS Assistant state dictionary

    Returns:
        The final result or None if not found
    """
    # Attempt to find the final result in the assistant state
    if "final_result" in state:
        return state["final_result"]

    # If not found directly, try to find it in nested structures
    for key, value in state.items():
        if isinstance(value, dict) and "response" in value:
            return value
        # Check if it's a model with a response attribute
        elif hasattr(value, "response"):
            return value
        # Check for response_generator output
        elif key == "response_generator_output" and value:
            return value

    # Additional checks for response in deeper nested structures
    for key, value in state.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if sub_key == "response" or (
                    isinstance(sub_value, dict) and "response" in sub_value
                ):
                    return sub_value

    return None


def extract_scores_from_results(
    results: List[Dict[str, Any]],
) -> List[Tuple[float, float, float, float]]:
    """Extract scores from evaluation results for analysis.

    Args:
        results: List of evaluation results

    Returns:
        List of score tuples (correctness, completeness, clarity, overall)
    """
    scores = []

    for result in results:
        evaluation = result.get("evaluation", {})
        score_dict = evaluation.get("scores", {})

        correctness = score_dict.get("correctness", 0)
        completeness = score_dict.get("completeness", 0)
        clarity = score_dict.get("clarity", 0)
        overall = evaluation.get("overall_score", 0)

        scores.append((correctness, completeness, clarity, overall))

    return scores
