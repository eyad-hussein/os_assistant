from typing import Any

from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import PromptTemplate

from os_assistant.config.settings import fixing_model
from os_assistant.parsers.json_cleaner import clean_and_parse_json
from os_assistant.pydantic_models.schemas import (
    CodeExecuteRequest,
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
code_execute_parser = PydanticOutputParser(pydantic_object=CodeExecuteRequest)

# Custom Prompt Template for OutputFixingParser
# This template instructs the LLM on how to fix malformed JSON.
output_fixing_template = """Instructions:
The following output was intended to be ONLY valid JSON conforming to the schema below, but it is malformed.
Please extract the valid JSON object from the output. Respond with ONLY the JSON object, nothing else.

Schema:
{instructions}

Malformed Output:
{completion}

Error Details:
{error}

IMPORTANT FIXES TO MAKE:
1. For PowerShell commands containing $_ variables, replace with $item or use {{$_}}
2. For any PowerShell commands with $ variables, escape them properly in JSON by doubling the $
3. Make sure all quotes within strings are properly escaped
4. Ensure all PowerShell pipeline operators (|) are preserved
5. Preserve all command line switches and parameters
6. For commands with multiple $ variables, make sure ALL of them are properly escaped with double $
7. For PowerShell hash tables, ensure proper JSON formatting for @{{}} syntax

Corrected JSON Output:
"""
output_fixing_prompt = PromptTemplate.from_template(output_fixing_template)

# Create fixed parsers with OutputFixingParser using the custom prompt
# These parsers attempt to automatically correct malformed JSON output from the LLM.
fixed_domain_analysis_parser = OutputFixingParser.from_llm(
    parser=domain_analysis_parser,
    llm=fixing_model,
    prompt=output_fixing_prompt,
    max_retries=10,  # Increased retries from default
)

fixed_query_type_parser = OutputFixingParser.from_llm(
    parser=query_type_parser,
    llm=fixing_model,
    prompt=output_fixing_prompt,
    max_retries=10,  # Increased retries
)

fixed_command_response_parser = OutputFixingParser.from_llm(
    parser=command_response_parser,
    llm=fixing_model,
    prompt=output_fixing_prompt,
    max_retries=10,  # Increased retries
)

fixed_info_response_parser = OutputFixingParser.from_llm(
    parser=info_response_parser,
    llm=fixing_model,
    prompt=output_fixing_prompt,
    max_retries=10,  # Increased retries
)

fixed_code_execute_parser = OutputFixingParser.from_llm(
    parser=code_execute_parser,
    llm=fixing_model,
    prompt=output_fixing_prompt,
    max_retries=10,  # Increased retries
)


def _extract_json_block(text: str) -> str | None:
    """Finds the first valid-looking JSON block ({} or []) in the text."""
    # Preprocess PowerShell commands to make them more JSON-friendly
    text = text.replace("$_", "$item")  # Replace $_ with $item to avoid JSON issues
    text = text.replace("$.", "$item.")  # Fix other PowerShell syntax

    # Handle other PowerShell variables by doubling the $
    import re

    # Find PowerShell variables but not already doubled ones
    text = re.sub(r"(?<!\$)\$([a-zA-Z_][a-zA-Z0-9_]*)", r"$$\1", text)

    # Handle PowerShell hash tables
    text = re.sub(r"@{([^}]*)}", r"{'\1'}", text)

    # Handle PowerShell arrays
    text = text.replace("@(", "[").replace(")", "]")

    # Better handling of PowerShell ForEach-Object and Where-Object
    text = text.replace("ForEach-Object", "ForEachObject")
    text = text.replace("Where-Object", "WhereObject")

    # Fix PowerShell parameter notation
    text = re.sub(r"-([a-zA-Z]+)\s+", r"'-\1': ", text)

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
        json_block = text[start_index : end_index + 1]
        return json_block
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

    # Check for empty content - this is a special case we need to handle
    if not content.strip():
        print("Warning: Empty or whitespace-only content received")

        # Create default minimal objects based on parser type
        if parser == info_response_parser:
            return InformationResponse(
                answer="I couldn't generate a specific answer based on the available information. Could you provide more details?",
                sources=["System analysis"],
                tool_breakdown=None,
                tool_results=None,
                tool_interpretation=None,
            )
        elif parser == command_response_parser:
            return CommandResponse(
                command="echo 'Unable to generate command'",
                what_command_does="I couldn't generate a specific command based on the available information.",
                security_notes="Please provide more specific details about what you want to accomplish.",
                tool_breakdown=None,
                tool_results=None,
                tool_interpretation=None,
            )
        # Add other parsers as needed

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
                    print(f"Parsing extracted JSON failed: {extract_error}")

                    # 4. Try using the clean_and_parse_json function
                    print("Attempting to clean and parse JSON...")
                    json_data = clean_and_parse_json(content)
                    if json_data:
                        try:
                            # Convert the dictionary to the expected Pydantic model
                            if parser == info_response_parser:
                                # Ensure minimum required fields are present
                                if "answer" not in json_data:
                                    json_data["answer"] = (
                                        "I found information related to your query, but couldn't format it properly."
                                    )
                                return InformationResponse.model_validate(json_data)
                            elif parser == command_response_parser:
                                # Ensure minimum required fields are present
                                if "command" not in json_data:
                                    json_data["command"] = (
                                        "echo 'Command generation incomplete'"
                                    )
                                if "what_command_does" not in json_data:
                                    json_data["what_command_does"] = (
                                        "The command was partially generated but couldn't be formatted correctly."
                                    )
                                return CommandResponse.model_validate(json_data)
                            else:
                                # Generic fallback
                                return parser.parse(str(json_data))
                        except Exception as e:
                            print(f"Failed to convert cleaned JSON to model: {str(e)}")

                    # If even extraction fails, use manual model creation as last resort
                    print("All JSON parsing methods failed. Creating default object...")
                    if parser == info_response_parser:
                        # Extract potential answer text from the content
                        import re

                        answer_match = re.search(r'"answer"\s*:\s*"([^"]+)"', content)
                        answer = (
                            answer_match.group(1)
                            if answer_match
                            else "I couldn't extract a specific answer from the available information."
                        )
                        return InformationResponse(
                            answer=answer,
                            sources=["System analysis"],
                            tool_breakdown=None,
                            tool_results=None,
                            tool_interpretation=None,
                        )
                    elif parser == command_response_parser:
                        # Extract potential command from the content
                        import re

                        cmd_match = re.search(r'"command"\s*:\s*"([^"]+)"', content)
                        command = (
                            cmd_match.group(1)
                            if cmd_match
                            else "echo 'Command extraction failed'"
                        )
                        return CommandResponse(
                            command=command,
                            what_command_does="The command was identified but couldn't be properly structured.",
                            security_notes="Use caution when executing this command as it was not properly validated.",
                            tool_breakdown=None,
                            tool_results=None,
                            tool_interpretation=None,
                        )
                    else:
                        # If we can't create a specific model, re-raise the fixing error
                        raise fix_error
            else:
                # If no JSON block could be extracted, try last resort methods
                print("Could not extract JSON block. Attempting emergency parsing...")

                # Try to create models from fragments of the content
                if parser == info_response_parser:
                    # Look for anything resembling an answer
                    answer = content
                    # Limit length
                    if len(answer) > 500:
                        answer = answer[:497] + "..."
                    return InformationResponse(
                        answer=answer,
                        sources=["Emergency extraction from malformed response"],
                        tool_breakdown=None,
                        tool_results=None,
                        tool_interpretation=None,
                    )
                elif parser == command_response_parser:
                    # Look for command-like content
                    import re

                    cmd_pattern = re.search(r"([\w-]+\s+[\w\s-]+)", content)
                    cmd = (
                        cmd_pattern.group(0)
                        if cmd_pattern
                        else "echo 'Command could not be extracted'"
                    )
                    return CommandResponse(
                        command=cmd,
                        what_command_does="This command was extracted from a malformed response and may not be complete.",
                        security_notes="Review this command carefully before execution as it was extracted from an invalid response.",
                        tool_breakdown=None,
                        tool_results=None,
                        tool_interpretation=None,
                    )
                else:
                    # Give up and raise the error
                    raise fix_error
