from typing import Any

from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import PromptTemplate

from ..config.settings import fixing_model
from ..models.schemas import (
    CommandResponse,
    DomainAnalysis,
    InformationResponse,
    QueryTypeResult,
)

# --- Parsers Setup ---

# Base Pydantic Parsers
domain_analysis_parser = PydanticOutputParser(pydantic_object=DomainAnalysis)
query_type_parser = PydanticOutputParser(pydantic_object=QueryTypeResult)
command_response_parser = PydanticOutputParser(pydantic_object=CommandResponse)
info_response_parser = PydanticOutputParser(pydantic_object=InformationResponse)

# Custom Prompt Template for OutputFixingParser
# This template instructs the LLM on how to fix malformed JSON.
output_fixing_template = """
Instructions:
The following output was intended to be ONLY valid JSON conforming to the schema below, but it is malformed.
Please extract the valid JSON object from the output. Respond with ONLY the JSON object, nothing else.

Schema:
{schema}

Malformed Output:
{output}

Corrected JSON Output:
"""
output_fixing_prompt = PromptTemplate.from_template(output_fixing_template)

# Create fixed parsers with OutputFixingParser using the custom prompt
# These parsers attempt to automatically correct malformed JSON output from the LLM.
fixed_domain_analysis_parser = OutputFixingParser.from_llm(
    parser=domain_analysis_parser,
    llm=fixing_model,  # Use the designated fixing model
    prompt=output_fixing_prompt.partial(
        schema=domain_analysis_parser.get_format_instructions()
    ),
)
fixed_query_type_parser = OutputFixingParser.from_llm(
    parser=query_type_parser,
    llm=fixing_model,
    prompt=output_fixing_prompt.partial(
        schema=query_type_parser.get_format_instructions()
    ),
)
fixed_command_response_parser = OutputFixingParser.from_llm(
    parser=command_response_parser,
    llm=fixing_model,
    prompt=output_fixing_prompt.partial(
        schema=command_response_parser.get_format_instructions()
    ),
)
fixed_info_response_parser = OutputFixingParser.from_llm(
    parser=info_response_parser,
    llm=fixing_model,
    prompt=output_fixing_prompt.partial(
        schema=info_response_parser.get_format_instructions()
    ),
)


def _extract_json_block(text: str) -> str | None:
    """Finds the first valid-looking JSON block ({} or []) in the text."""
    start_brace = text.find("{")
    start_bracket = text.find("[")

    # Determine the start index of the first JSON object or array
    if start_brace == -1 and start_bracket == -1:
        return None  # No JSON structure found

    start_index = -1
    if start_brace != -1 and start_bracket != -1:
        start_index = min(start_brace, start_bracket)
    elif start_brace != -1:
        start_index = start_brace
    else:
        start_index = start_bracket

    # Determine if it's an object or array to find the correct closing character
    is_object = start_index == start_brace
    open_char = "{" if is_object else "["
    close_char = "}" if is_object else "]"

    balance = 0
    end_index = -1
    in_string = False
    escaped = False

    # Iterate through the string to find the matching closing character
    for i in range(start_index, len(text)):
        char = text[i]

        if in_string:
            # Handle characters within strings
            if char == '"' and not escaped:
                in_string = False
            elif char == "\\" and not escaped:
                escaped = True
            else:
                escaped = False
        else:
            # Handle characters outside strings
            if char == '"':
                in_string = True
                escaped = False
            elif char == open_char:
                balance += 1
            elif char == close_char:
                balance -= 1
                if balance == 0:
                    end_index = i
                    break  # Found the end of the JSON block

    if end_index != -1:
        # Return the extracted JSON block
        return text[start_index : end_index + 1]
    else:
        # Return None if a complete block wasn't found
        return None


def parse_with_fix_and_extract(
    content: Any, parser: PydanticOutputParser, fixer: OutputFixingParser
) -> Any:
    """
    Attempts to parse LLM output, falling back to fixing and then simple extraction.

    Args:
        content: The raw output from the LLM (usually a string).
        parser: The primary PydanticOutputParser.
        fixer: The OutputFixingParser used as a fallback.

    Returns:
        The parsed Pydantic object or raises an exception if all attempts fail.

    Raises:
        OutputParserException: If parsing, fixing, and extraction all fail.
    """
    # Ensure content is a string for parsing attempts
    if not isinstance(content, str):
        # Try converting common types like AIMessage content
        try:
            content = str(content.content)
        except AttributeError:
            content = str(content)  # Fallback to generic string conversion

    try:
        # 1. Try direct parsing first
        return parser.parse(content)
    except OutputParserException as direct_error:
        print(f"Direct parsing failed: {direct_error}. Attempting fixing...")
        try:
            # 2. If direct fails, try the fixing parser
            return fixer.parse(content)
        except OutputParserException as fix_error:
            print(
                f"Fixing parser failed: {fix_error}. Attempting simple JSON extraction..."
            )
            # 3. If fixing fails, try simple extraction of the first JSON block
            extracted_json = _extract_json_block(content)
            if extracted_json:
                try:
                    # Try parsing the extracted block
                    print("Extracted JSON block, attempting to parse it...")
                    return parser.parse(extracted_json)
                except OutputParserException as extract_error:
                    # If even extraction fails, raise the fixing error (often more informative)
                    print(f"Parsing extracted JSON failed: {extract_error}")
                    raise fix_error  # Raise the error from the fixing attempt
            else:
                # If no JSON block could be extracted, raise the fixing error
                print("Could not extract JSON block after fixing failed.")
                raise fix_error  # Raise the error from the fixing attempt
