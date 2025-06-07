import os
import time
from typing import Any, Dict, Tuple

import yaml
from langchain.schema import HumanMessage
from langchain_ollama import ChatOllama

from os_assistant.config.settings import MODEL_BASE_URL, MODEL_NAME


class LLMJudge:
    """Judge that uses an LLM to evaluate OS Assistant responses."""

    def __init__(self, model_name: str | None = None, base_url: str | None = None):
        """Initialize the LLM judge.

        Args:
            model_name: Name of the LLM model to use, defaults to the one in settings
            base_url: Base URL for the Ollama API, defaults to the one in settings
        """
        self.model_name = model_name or MODEL_NAME
        self.base_url = base_url or MODEL_BASE_URL
        self.model = ChatOllama(
            model=self.model_name, temperature=0, base_url=self.base_url
        )

        # Define enhanced evaluation prompt with detailed rating explanations
        self.evaluation_prompt = """
        Evaluate the generated answer based on the following criteria, rating each on a scale of 1-5 where 5 is best:

        Correctness (1-5):
        - 1: Completely incorrect, contains false information, or is harmful
        - 2: Mostly incorrect with some accurate elements
        - 3: Partially correct but with significant errors or omissions
        - 4: Mostly correct with minor inaccuracies
        - 5: Completely correct and accurate

        Completeness (1-5):
        - 1: Completely incomplete, missing almost all required information
        - 2: Severely lacking, addresses only a small portion of the question
        - 3: Partially complete, covers main points but misses important details
        - 4: Mostly complete with minor omissions
        - 5: Fully complete, addresses all aspects of the question

        Question: {question}
        
        Expected (Model) Answer: {expected_response}
        
        Actual Answer: {actual_response}
        
        Query Type: {query_type}
        
        Provide your evaluation in the following JSON format:
        {{
            "scores": {{
                "correctness": [1-5],
                "correctness_explanation": "[detailed explanation why this correctness score was assigned, referencing the 1-5 scale definition]",
                "completeness": [1-5],
                "completeness_explanation": "[detailed explanation why this completeness score was assigned, referencing the 1-5 scale definition]"
            }},
            "overall_score": [calculated average of above scores],
            "reasoning": "[detailed explanation of your overall evaluation, including specific strengths and weaknesses]"
        }}
        
        Make sure to include the explanation directly after each score in the JSON structure, explaining why you assigned that specific rating.
        """

    def evaluate(
        self,
        question: str,
        expected_response: str,
        actual_response: Dict[str, Any],
        query_type: str,
    ) -> Tuple[Dict, float]:
        """Evaluate the assistant's response against the expected response.

        Args:
            question: The original question
            expected_response: The expected response from the dataset
            actual_response: The actual response from the assistant (as dict)
            query_type: Type of query ('command' or 'information')

        Returns:
            Tuple of (evaluation results dictionary, evaluation latency in ms)
        """
        # Format the actual response based on type
        formatted_actual = self._format_response(actual_response, query_type)

        # Format the prompt
        prompt = self.evaluation_prompt.format(
            question=question,
            expected_response=expected_response,
            actual_response=formatted_actual,
            query_type=query_type,
        )

        # Invoke the LLM with latency tracking
        messages = [HumanMessage(content=prompt)]
        start_time = time.time()
        response = self.model.invoke(messages)
        end_time = time.time()

        # Calculate latency in milliseconds
        latency_ms = (end_time - start_time) * 1000

        # Parse the response
        from ..utils.parser import parse_evaluation_result

        evaluation_result = parse_evaluation_result(response.content)

        return evaluation_result, latency_ms

    def _format_response(self, response: Dict[str, Any], query_type: str) -> str:
        """Format the assistant's response for evaluation.

        Args:
            response: The response dictionary from the assistant
            query_type: Type of query ('command' or 'information')

        Returns:
            Formatted response as a string
        """
        if query_type == "command":
            # Format command response
            command = response.get("command", "")
            explanation = response.get("explanation", "")
            security_notes = response.get("security_notes", "")

            formatted = f"Command: {command}\n\nExplanation: {explanation}"
            if security_notes:
                formatted += f"\n\nSecurity Notes: {security_notes}"

        else:
            # Format information response
            answer = response.get("answer", "")
            sources = response.get("sources", [])
            sources_str = ", ".join(sources) if sources else "No sources provided"

            formatted = f"Information: {answer}\n\nSources: {sources_str}"

        return formatted
