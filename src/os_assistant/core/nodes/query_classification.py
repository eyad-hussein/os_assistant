from langchain.schema import HumanMessage

from os_assistant.core.nodes.helpers import build_combined_context
from os_assistant.core.state import AssistantState
from os_assistant.parsers.setup import (
    fixed_query_type_parser,
    parse_with_fix_and_extract,
    query_type_parser,
)
from os_assistant.prompts.prompt_loader import load_prompt
from os_assistant.pydantic_models.schemas import QueryTypeResult
from os_assistant.utils.model_factory import model


def query_classifier_node(state: AssistantState) -> AssistantState:
    """Classify the query type (command or information)"""
    print("\nNODE: query_classifier_node")
    print("\nClassifying query type...")

    # Use helper function to build combined context
    combined_context = build_combined_context(state)

    # Load prompt from YAML
    query_classifier_yaml = load_prompt("query_classifier_node")

    # Format the prompt with required variables
    prompt = query_classifier_yaml["prompt"].format(
        prompt=state["prompt"],
        combined_context=combined_context,
        format_instructions=query_type_parser.get_format_instructions(),
    )

    messages = [HumanMessage(content=prompt)]
    content = model.invoke(messages)

    try:
        # Use the helper function for parsing attempts
        query_type = parse_with_fix_and_extract(
            content, query_type_parser, fixed_query_type_parser
        )

        # Ensure the result is a Pydantic model instance
        if not isinstance(query_type, QueryTypeResult):
            query_type = QueryTypeResult.model_validate(query_type)

        state["query_type"] = query_type

        print(f"Query classified as: {query_type.query_type}")
        print(f"Reasoning: {query_type.reasoning}")

    except Exception as e:
        print(f"Error classifying query: {str(e)}")
        # Fallback to information type
        fallback_query_type = QueryTypeResult(
            query_type="information",
            reasoning=f"Fallback: defaulting to information type due to classification error for query: '{state['prompt']}'",
            confidence=0.5,
        )
        state["query_type"] = fallback_query_type

    return state
