import json
import re
from typing import Any

from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from os_assistant.utils import LOGGER

from ..config.config import LLM_MODEL, LLM_TEMPERATURE, OLLAMA_BASE_URL
from ..core.models import CodeAnalysis


def create_code_analysis_parser() -> PydanticOutputParser[CodeAnalysis]:
    """Create a parser for the CodeAnalysis model"""
    return PydanticOutputParser(pydantic_object=CodeAnalysis)


def get_parsing_instructions() -> str:
    """Get parsing instructions for the LLM"""
    parser = create_code_analysis_parser()
    return parser.get_format_instructions()


def ensure_string(message: Any) -> str:
    """Convert any message object to a string"""
    if isinstance(message, AIMessage):
        return str(message.content)
    elif hasattr(message, "content"):
        return str(message.content)
    return str(message)


def create_output_fixing_prompt():
    """Create a prompt template for fixing malformed JSON"""
    template = """
    Instructions:
    The following output was intended to be valid JSON conforming to the schema below, but it is malformed.
    Please extract the valid JSON object from the output. Respond with ONLY the JSON object, nothing else.

    Schema:
    {instructions}

    Malformed Output:
    {completion}

    Error Details:
    {error}

    IMPORTANT FIXES TO MAKE:
    1. Make sure all quotes within strings are properly escaped
    2. Ensure all fields are properly formatted
    3. Fix any syntax errors in the JSON
    4. Maintain the original code structure as much as possible

    Corrected JSON Output:
    """
    return PromptTemplate.from_template(template)


def create_fixing_parser(parser):
    """Create an OutputFixingParser with increased retries"""
    # Create LLM directly instead of importing from agents.py to break circular dependency
    llm = ChatOllama(
        model=LLM_MODEL, temperature=LLM_TEMPERATURE, base_url=OLLAMA_BASE_URL
    )
    prompt = create_output_fixing_prompt()
    return OutputFixingParser.from_llm(
        parser=parser,
        llm=llm,
        prompt=prompt,
        max_retries=5,  # Increased retries
    )


def extract_json_manually(text: str) -> dict | None:
    """Manually extract JSON from text that might contain markdown or other content"""
    try:
        # Ensure we're working with a string
        text = ensure_string(text)

        # Remove "json" prefix if present (common LLM response pattern)
        if text.lstrip().startswith("json"):
            text = re.sub(r"^\s*json\s*", "", text)

        # Find the first opening brace
        start_idx = text.find("{")
        if start_idx == -1:
            return {}

        # Find the matching closing brace
        brace_count = 0
        end_idx = -1
        for i in range(start_idx, len(text)):
            if text[i] == "{":
                brace_count += 1
            elif text[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break

        if end_idx == -1:
            return {}

        # Extract the JSON string
        json_str = text[start_idx:end_idx]

        # Parse the JSON
        try:
            data = json.loads(json_str)

            # Handle nested code structure {"code": {"python_code": "..."}}
            if isinstance(data.get("code"), dict) and "python_code" in data["code"]:
                python_code = data["code"]["python_code"]
                # Replace nested object with the extracted code string
                data["code"] = python_code
                LOGGER.info(
                    "Successfully extracted nested python_code from JSON structure"
                )

            return data
        except json.JSONDecodeError:
            # Try to clean and fix common JSON issues
            cleaned_json = json_str.replace('\\"', '"').replace("\\n", "\n")
            try:
                data = json.loads(cleaned_json)
                # Handle nested code structure again after cleaning
                if isinstance(data.get("code"), dict) and "python_code" in data["code"]:
                    data["code"] = data["code"]["python_code"]
                return data
            except json.JSONDecodeError:
                # Final attempt with regex-based extraction
                return extract_code_fields_with_regex(text)
    except Exception as e:
        LOGGER.error(f"Manual JSON extraction failed: {str(e)}")

    return {}


def extract_code_fields_with_regex(text: str) -> dict:
    """Extract code and other fields using regex patterns when JSON parsing fails"""
    result = {}

    # Try to extract the code field - handle triple quotes or other formatting
    code_pattern = r'"code"\s*:\s*(?:{[\s\S]*?"python_code"\s*:\s*(?:"""|\"{3})([\s\S]*?)(?:"""|\"{3})|"((?:\\.|[^"\\])*)"|```python\s*([\s\S]*?)```)'
    code_match = re.search(code_pattern, text)

    if code_match:
        # Get the first non-None group - that's our code
        for group in code_match.groups():
            if group is not None:
                result["code"] = group
                break
    else:
        # Fallback to looking for Python code blocks
        code_blocks = re.findall(r"```python\s*([\s\S]*?)```", text)
        if code_blocks:
            result["code"] = code_blocks[0]

    # Extract dangerous level
    danger_match = re.search(r'"dangerous"\s*:\s*(\d+)', text)
    if danger_match:
        try:
            result["dangerous"] = int(danger_match.group(1))
        except ValueError:
            result["dangerous"] = 1
    else:
        result["dangerous"] = 1

    # Extract reason
    reason_match = re.search(r'"reason"\s*:\s*"((?:\\.|[^"\\])*)"', text)
    if reason_match:
        result["reason"] = reason_match.group(1)
    else:
        result["reason"] = "Extracted with regex patterns"

    return result


def fix_incomplete_json(json_str: str) -> str:
    """Fix common JSON issues like unclosed braces and quotes"""
    # Count opening and closing braces
    open_braces = json_str.count("{")
    close_braces = json_str.count("}")

    # Add missing closing braces
    if open_braces > close_braces:
        json_str += "}" * (open_braces - close_braces)

    # Check for unclosed quotes in key-value pairs
    # This is a simplified approach - might not work for all cases
    key_value_pattern = r'"([^"]+)"\s*:\s*"([^"]*)'
    matches = re.finditer(key_value_pattern, json_str)

    fixed_json = json_str
    for match in matches:
        if not re.search(f'"{match.group(1)}"\\s*:\\s*"[^"]*"', json_str):
            # This quote was never closed
            replacement = f'"{match.group(1)}": "{match.group(2)}"'
            fixed_json = fixed_json.replace(match.group(0), replacement)

    return fixed_json


def parse_structured_output(response_text, model_class):
    """Parse structured output from LLM response with multiple fallback mechanisms"""
    # Convert AIMessage to string if necessary
    response_text = ensure_string(response_text)

    # Create parsers
    parser = PydanticOutputParser(pydantic_object=model_class)
    fixing_parser = create_fixing_parser(parser)

    # First try to parse the response as is
    try:
        return parser.parse(response_text)
    except OutputParserException as e:
        LOGGER.error(f"Standard parsing failed: {str(e)}")

        # Second try with the fixing parser - give it multiple attempts
        try:
            LOGGER.info("Attempting to fix malformed output...")
            # The fixing parser will try up to 5 times (configured in create_fixing_parser)
            fixed_result = fixing_parser.parse(response_text)
            LOGGER.info("Successfully fixed and parsed the output!")
            return fixed_result
        except OutputParserException as e2:
            LOGGER.error(f"Fixing parser failed after multiple attempts: {str(e2)}")

            # Third try to manually extract JSON as a last resort
            LOGGER.warning(
                "All structured parsing attempts failed. Trying manual JSON extraction..."
            )
            json_data = extract_json_manually(response_text)

            if json_data and "code" in json_data:
                try:
                    # Convert to Pydantic model
                    if model_class == CodeAnalysis:
                        return CodeAnalysis(
                            code=json_data.get("code", ""),
                            dangerous=json_data.get("dangerous", 1),
                            reason=json_data.get(
                                "reason", "Extracted manually from response"
                            ),
                        )
                except Exception as e3:
                    LOGGER.error(
                        f"Failed to create model from extracted JSON: {str(e3)}"
                    )

            # Final fallback: extract code and create a basic analysis
            code = extract_code_from_markdown(response_text)
            return CodeAnalysis(
                code=code,
                dangerous=1,
                reason="Parser couldn't extract danger assessment, using default safe level.",
            )


def extract_json_from_text(text) -> None | str:
    """Extract JSON from text by finding sections between curly braces"""
    # Convert to string if needed
    text = ensure_string(text)

    start = text.find("{")
    if start == -1:
        return None

    # Find matching closing brace
    open_count = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            open_count += 1
        elif text[i] == "}":
            open_count -= 1
            if open_count == 0:
                return text[start : i + 1]

    return None


def extract_code_from_markdown(text) -> str:
    """Extract code from markdown code blocks"""
    # Convert to string if needed
    text = ensure_string(text)

    # First try to extract Python code blocks
    if "```python" in text and "```" in text:
        code_blocks = text.split("```python")[1:]
        for block in code_blocks:
            if "```" in block:
                # Get the first code block
                return block.split("```")[0].strip()

    # If no Python blocks, try any code blocks
    elif "```" in text:
        code_blocks = text.split("```")[
            1::2
        ]  # Take odd-indexed elements (inside blocks)
        if code_blocks:
            return code_blocks[0].strip()

    # Strip any JSON markers that might remain in the text
    cleaned_text = re.sub(r'^\s*{\s*"code"\s*:\s*"""', "", text)
    cleaned_text = re.sub(r'"""\s*,\s*"dangerous".+', "", cleaned_text)

    return cleaned_text
