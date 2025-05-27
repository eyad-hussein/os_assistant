import os
from typing import Any, Dict

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
        self.model = ChatOllama(model=self.model_name, base_url=self.base_url)

        # Load evaluation prompts
        prompts_path = "src/os_assistant/prompts/evaluation.yaml"
        with open(prompts_path, "r", encoding="utf-8") as f:
            self.prompts = yaml.safe_load(f)

    def evaluate(
        self,
        question: str,
        expected_response: str,
        actual_response: Dict[str, Any],
        query_type: str,
    ) -> Dict:
        """Evaluate the assistant's response against the expected response.

        Args:
            question: The original question
            expected_response: The expected response from the dataset
            actual_response: The actual response from the assistant (as dict)
            query_type: Type of query ('command' or 'information')

        Returns:
            Dictionary containing the evaluation results with scores
        """
        # Format the actual response based on type
        formatted_actual = self._format_response(actual_response, query_type)

        # Select the appropriate prompt based on query type
        if query_type == "command":
            prompt_template = self.prompts.get("command_evaluation_prompt")
        else:
            prompt_template = self.prompts.get("information_evaluation_prompt")

        if not prompt_template:
            # Fallback to general prompt if specific one not found
            prompt_template = self.prompts.get("general_evaluation_prompt")

        # Format the prompt
        prompt = prompt_template.format(
            question=question,
            expected_response=expected_response,
            actual_response=formatted_actual,
            query_type=query_type,
        )

        # Invoke the LLM
        messages = [HumanMessage(content=prompt)]
        response = self.model.invoke(messages)

        # Parse the response
        from ..utils.parser import parse_evaluation_result

        return parse_evaluation_result(response.content)

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
