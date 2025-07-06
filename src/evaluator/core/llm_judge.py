import os
import time
from typing import Any

import yaml
from langchain.schema import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from evaluator.config.config import (
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    PROMPTS_DIR,
    get_metric_weights,
)


class LLMJudge:
    """Judge that uses an LLM to evaluate OS Assistant responses."""

    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
        temperature: float = LLM_TEMPERATURE,
    ):
        """Initialize the LLM judge.

        Args:
            model_name: Name of the LLM model to use, defaults to the one in config
            base_url: Base URL for the Ollama API, defaults to the one in config
            temperature: Temperature for the LLM, defaults to the one in config
        """
        self.model_name = model_name or LLM_MODEL
        self.base_url = base_url or LLM_BASE_URL
        self.model = ChatOllama(
            model=self.model_name, temperature=temperature, base_url=self.base_url
        )

        # Load evaluation prompts from YAML files
        self.prompts = self._load_evaluation_prompts()

        # Get metric weights from config
        self.metric_weights = get_metric_weights()

    def _load_evaluation_prompts(self) -> dict[str, dict[str, str]]:
        """Load evaluation prompts from YAML files.

        Returns:
            Dictionary of prompts by evaluation type
        """
        prompts = {}

        # List of prompt files to load
        prompt_files = [
            "correctness_evaluator.yaml",
            "completeness_evaluator.yaml",
            "clarity_evaluator.yaml",
        ]

        for prompt_file in prompt_files:
            try:
                file_path = os.path.join(PROMPTS_DIR, prompt_file)
                if os.path.exists(file_path):
                    with open(file_path, encoding="utf-8") as f:
                        prompt_data = yaml.safe_load(f)
                        prompt_type = prompt_file.split("_")[0]
                        prompts[prompt_type] = prompt_data
                else:
                    print(f"Warning: Prompt file not found: {file_path}")
            except Exception as e:
                print(f"Error loading prompt file {prompt_file}: {str(e)}")

        return prompts

    def evaluate(
        self,
        question: str,
        expected_response: str,
        actual_response: dict[str, Any],
        query_type: str,
    ) -> tuple[dict, dict[str, float]]:
        """Evaluate the assistant's response against the expected response.

        Args:
            question: The original question
            expected_response: The expected response from the dataset
            actual_response: The actual response from the assistant (as dict)
            query_type: Type of query ('command' or 'information')

        Returns:
            Tuple of (evaluation results dictionary, latency metrics dict)
        """
        # Format the actual response based on type
        formatted_actual = self._format_response(actual_response, query_type)

        # Track latencies for each dimension
        latency_metrics = {}
        evaluation_results = {}

        # Evaluate each dimension separately for more focused assessments
        dimensions = list(self.metric_weights.keys())

        for dimension in dimensions:
            if dimension in self.prompts:
                prompt_data = self.prompts[dimension]

                # Create prompt based on dimension
                dimension_prompt = self._create_evaluation_prompt(
                    prompt_data,
                    question,
                    expected_response,
                    formatted_actual,
                    query_type,
                    dimension,
                )

                # Evaluate this dimension
                start_time = time.time()
                dimension_result = self._evaluate_dimension(dimension_prompt)
                end_time = time.time()

                # Track latency for this dimension
                latency_ms = (end_time - start_time) * 1000
                latency_metrics[f"{dimension}_evaluation_ms"] = latency_ms

                # Store result
                evaluation_results[dimension] = dimension_result
            else:
                print(f"Warning: No prompt found for dimension '{dimension}'")
                evaluation_results[dimension] = {
                    "score": 3,
                    "explanation": "Not evaluated",
                }

        # Calculate overall score - weighted average
        total_weight = sum(self.metric_weights.values())
        overall_score = 0.0

        for dimension, result in evaluation_results.items():
            if dimension in self.metric_weights:
                score = result.get("score", 3)
                weight = self.metric_weights[dimension] / total_weight
                overall_score += score * weight

        # Total latency
        total_latency_ms = sum(latency_metrics.values())
        latency_metrics["total_evaluation_ms"] = total_latency_ms

        # Combine results into final evaluation
        final_result = {
            "scores": {
                "correctness": evaluation_results["correctness"].get("score", 3),
                "correctness_explanation": evaluation_results["correctness"].get(
                    "explanation", ""
                ),
                "completeness": evaluation_results["completeness"].get("score", 3),
                "completeness_explanation": evaluation_results["completeness"].get(
                    "explanation", ""
                ),
                "clarity": evaluation_results["clarity"].get("score", 3),
                "clarity_explanation": evaluation_results["clarity"].get(
                    "explanation", ""
                ),
            },
            "overall_score": overall_score,
            "reasoning": self._generate_combined_reasoning(evaluation_results),
        }

        return final_result, latency_metrics

    def _create_evaluation_prompt(
        self,
        prompt_data: dict[str, str],
        question: str,
        expected_response: str,
        formatted_actual: str,
        query_type: str,
        dimension: str,
    ) -> list[dict[str, str]]:
        """Create an evaluation prompt for a specific dimension.

        Args:
            prompt_data: The prompt data from YAML
            question: The original question
            expected_response: The expected response
            formatted_actual: The formatted actual response
            query_type: Type of query ('command' or 'information')
            dimension: Evaluation dimension (correctness, completeness, etc.)

        Returns:
            List of messages for LLM
        """
        system_message = prompt_data.get(
            "system_message",
            f"You are an expert evaluator for the {dimension} of Windows 10 assistant responses.",
        )
        prompt_template = prompt_data.get("prompt", "")

        # Format the prompt with the query
        formatted_prompt = prompt_template.format(
            question=question,
            expected_response=expected_response,
            actual_response=formatted_actual,
            query_type=query_type,
            dimension=dimension.upper(),
        )

        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=formatted_prompt),
        ]

        return messages

    def _evaluate_dimension(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Evaluate a specific dimension with the LLM.

        Args:
            messages: The messages to send to the LLM

        Returns:
            The evaluation result for this dimension
        """
        response = self.model.invoke(messages)
        response_text = response.content

        # Extract score and explanation
        score = self._extract_score(response_text)
        explanation = self._extract_explanation(response_text)

        return {
            "score": score,
            "explanation": explanation,
        }

    def _extract_score(self, response_text: str) -> float:
        """Extract the score from the LLM response with improved patterns.

        Args:
            response_text: The raw response text

        Returns:
            The extracted score as a float
        """
        try:
            import re

            # First, search for explicit score patterns with more variations
            explicit_patterns = [
                # Look for dimension-specific scores
                r"CORRECTNESS SCORE:?\s*(\d+(?:\.\d+)?)",
                r"COMPLETENESS SCORE:?\s*(\d+(?:\.\d+)?)",
                r"CLARITY SCORE:?\s*(\d+(?:\.\d+)?)",
                # General score formats
                r"(?:SCORE|Score|score):?\s*(\d+(?:\.\d+)?)",
                r"(\d+(?:\.\d+)?)\s*\/\s*5",
                r"(\d+(?:\.\d+)?)\s*out of\s*5",
                # Score with labels
                r"(?:give|rate|assign|award)(?:s|ed)? (?:a|an|the)? score (?:of)? (\d+(?:\.\d+)?)",
                # Additional formats with rating terminology
                r"rating:?\s*(\d+(?:\.\d+)?)",
                r"grade:?\s*(\d+(?:\.\d+)?)",
                r"assessment:?\s*(\d+(?:\.\d+)?)",
            ]

            # Try the explicit patterns first
            for pattern in explicit_patterns:
                match = re.search(pattern, response_text, re.IGNORECASE)
                if match:
                    try:
                        score = float(match.group(1))
                        # Validate the score is within range
                        if 1 <= score <= 5:
                            return score
                        print(f"Found score {score} but it's outside valid range 1-5")
                    except ValueError:
                        continue

            # Secondary pass: Look for standalone numbers that might be scores
            # Focus on portions of text that might contain scores
            score_sections = [
                "CORRECTNESS SCORE",
                "COMPLETENESS SCORE",
                "CLARITY SCORE",
                "SCORE",
                "Score",
                "score",
                "Rating",
                "RATING",
            ]

            for section in score_sections:
                section_match = re.search(
                    f"{section}.*?(\d+(?:\.\d+)?)",
                    response_text,
                    re.IGNORECASE | re.DOTALL,
                )
                if section_match:
                    try:
                        score = float(section_match.group(1))
                        if 1 <= score <= 5:
                            return score
                    except ValueError:
                        continue

            # Look for any numbers between 1 and 5 that could be scores near key phrases
            number_near_score = re.search(
                r"(?:score|rating|grade|give|assign|award).{0,20}?(\d+(?:\.\d+)?)",
                response_text,
                re.IGNORECASE,
            )
            if number_near_score:
                try:
                    score = float(number_near_score.group(1))
                    if 1 <= score <= 5:
                        return score
                except ValueError:
                    pass

            # If we reach here, we couldn't find a valid score
            print("No valid score found in response, defaulting to 3.0")
            print(f"Response excerpt: {response_text[:200]}...")
            return 3.0
        except Exception as e:
            print(f"Error extracting score: {str(e)}")
            return 3.0

    def _extract_explanation(self, response_text: str) -> str:
        """Extract the explanation from the LLM response with improved patterns.

        Args:
            response_text: The raw response text

        Returns:
            The extracted explanation as a string
        """
        try:
            import re

            # Look for explanation sections with more variations
            explanation_patterns = [
                r"JUSTIFICATION:?\s*([\s\S]+?)(?:\n\n|\n[A-Z][A-Z]+:|\Z)",
                r"EXPLANATION:?\s*([\s\S]+?)(?:\n\n|\n[A-Z][A-Z]+:|\Z)",
                r"REASONING:?\s*([\s\S]+?)(?:\n\n|\n[A-Z][A-Z]+:|\Z)",
                r"ANALYSIS:?\s*([\s\S]+?)(?:\n\n|\n[A-Z][A-Z]+:|\Z)",
                r"RATIONALE:?\s*([\s\S]+?)(?:\n\n|\n[A-Z][A-Z]+:|\Z)",
            ]

            for pattern in explanation_patterns:
                matches = re.search(pattern, response_text, re.IGNORECASE)
                if matches:
                    explanation = matches.group(1).strip()
                    # Only return if we got something substantial
                    if len(explanation) > 20:  # Minimum length to be considered valid
                        return explanation

            # If no specific explanation section found, look for any substantial paragraph
            paragraphs = re.split(r"\n\n+", response_text)
            for para in paragraphs:
                # Skip short lines and lines that look like headers
                if len(para.strip()) > 100 and not re.match(
                    r"^[A-Z\s]+:$", para.strip()
                ):
                    return para.strip()

            # If all else fails, return most of the response, removing any obvious headers
            cleaned_text = re.sub(r"^[A-Z\s]+:", "", response_text, flags=re.MULTILINE)
            return cleaned_text.strip()
        except Exception as e:
            print(f"Error extracting explanation: {str(e)}")
            return response_text

    def _generate_combined_reasoning(self, results: dict[str, dict[str, Any]]) -> str:
        """Generate a combined reasoning from all dimensions.

        Args:
            results: The results from all dimensions

        Returns:
            A combined reasoning string
        """
        combined = "Overall Assessment:\n\n"

        # Add each dimension's explanation
        for dimension in ["correctness", "completeness", "clarity"]:
            if dimension in results:
                score = results[dimension].get("score", 3)
                explanation = results[dimension].get("explanation", "")

                # Add a summary line from the explanation (first sentence or two)
                summary = (
                    explanation.split(".", 1)[0] + "."
                    if "." in explanation
                    else explanation
                )
                combined += f"{dimension.capitalize()} ({score}/5): {summary}\n\n"

        # Add a conclusion
        strongest = max(results.items(), key=lambda x: x[1].get("score", 0))[0]
        weakest = min(results.items(), key=lambda x: x[1].get("score", 0))[0]

        combined += f"The response's strongest aspect is {strongest}, while {weakest} could be improved."

        return combined

    def _format_response(self, response: dict[str, Any] | Any, query_type: str) -> str:
        """Format the assistant's response for evaluation.

        Args:
            response: The response from the assistant (dict or Pydantic model)
            query_type: Type of query ('command' or 'information')

        Returns:
            Formatted response as a string
        """
        if query_type == "command":
            # Format command response
            command = self._get_attribute_safely(response, "command", "")
            what_command_does = self._get_attribute_safely(
                response, "what_command_does", ""
            )
            # Check for legacy explanation field if what_command_does is empty
            if not what_command_does:
                what_command_does = self._get_attribute_safely(
                    response, "explanation", ""
                )
            security_notes = self._get_attribute_safely(response, "security_notes", "")
            tool_breakdown = self._get_attribute_safely(response, "tool_breakdown", "")
            tool_results = self._get_attribute_safely(response, "tool_results", "")
            tool_interpretation = self._get_attribute_safely(
                response, "tool_interpretation", ""
            )

            formatted = (
                f"Command: {command}\n\nWhat this command does: {what_command_does}"
            )
            if security_notes:
                formatted += f"\n\nSecurity Notes: {security_notes}"
            if tool_breakdown:
                formatted += f"\n\nTool Usage: {tool_breakdown}"
            if tool_results:
                formatted += f"\n\nTool Results: {tool_results}"
            if tool_interpretation:
                formatted += f"\n\nTool Interpretation: {tool_interpretation}"

        else:
            # Format information response
            answer = self._get_attribute_safely(response, "answer", "")
            sources = self._get_attribute_safely(response, "sources", [])
            tool_breakdown = self._get_attribute_safely(response, "tool_breakdown", "")
            tool_results = self._get_attribute_safely(response, "tool_results", "")
            tool_interpretation = self._get_attribute_safely(
                response, "tool_interpretation", ""
            )

            sources_str = ", ".join(sources) if sources else "No sources provided"
            formatted = f"Information: {answer}\n\nSources: {sources_str}"
            if tool_breakdown:
                formatted += f"\n\nTool Usage: {tool_breakdown}"
            if tool_results:
                formatted += f"\n\nTool Results: {tool_results}"
            if tool_interpretation:
                formatted += f"\n\nTool Interpretation: {tool_interpretation}"

        return formatted

    def _get_attribute_safely(self, obj: Any, attr: str, default: Any = None) -> Any:
        """Safely get an attribute from an object, whether it's a dict or a model.

        Args:
            obj: The object to get the attribute from
            attr: The attribute name to get
            default: The default value to return if the attribute is not found

        Returns:
            The attribute value or the default
        """
        if obj is None:
            return default

        # If it's a dictionary, use get method
        if isinstance(obj, dict):
            return obj.get(attr, default)

        # If it's a model with attributes, use getattr
        if hasattr(obj, attr):
            return getattr(obj, attr)

        # If it's a model with model_dump method (Pydantic v2+)
        if hasattr(obj, "model_dump"):
            return obj.model_dump().get(attr, default)

        # If it's a model with dict method (Pydantic v1)
        if hasattr(obj, "dict"):
            return obj.dict().get(attr, default)

        # If all else fails, return the default
        return default
