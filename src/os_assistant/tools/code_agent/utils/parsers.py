import re
import json
from typing import Any, Optional, Dict

from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain_core.messages import AIMessage
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from ..core.models import CodeAnalysis
from ..config.config import LLM_MODEL, LLM_TEMPERATURE, OLLAMA_BASE_URL


def create_code_analysis_parser():
    """Create a parser for the CodeAnalysis model"""
    return PydanticOutputParser(pydantic_object=CodeAnalysis)


def get_parsing_instructions():
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
    {schema}

    Malformed Output:
    {output}

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


def extract_json_manually(text: str) -> Optional[Dict]:
    """Manually extract JSON from text when parsing fails"""
    # Ensure we have a string to work with
    text = ensure_string(text)

    # Method 1: Try to find well-formed JSON between triple backticks
    json_pattern = r"```(?:json)?\s*(\{[\s\S]*?\})\s*```"
    json_matches = re.findall(json_pattern, text)

    for json_str in json_matches:
        try:
            parsed_json = json.loads(json_str)
            if isinstance(parsed_json, dict) and "code" in parsed_json:
                return parsed_json
        except json.JSONDecodeError:
            continue

    # Method 2: Try to reconstruct malformed JSON from keys and values
    # Look for code, dangerous and reason fields
    code_pattern = r'"code"\s*:\s*"((?:\\.|[^"\\])*)"'
    # If the above fails, try with triple quotes
    triple_code_pattern = r'"code"\s*:\s*(?:"""|\"{3})([\s\S]*?)(?:"""|\"{3})'
    dangerous_pattern = r'"dangerous"\s*:\s*(\d+)'
    reason_pattern = r'"reason"\s*:\s*"((?:\\.|[^"\\])*)"'

    # Extract code
    code_match = re.search(triple_code_pattern, text)
    if not code_match:
        code_match = re.search(code_pattern, text)

    if code_match:
        code = code_match.group(1)

        # Extract dangerous level
        dangerous = 1  # Default value
        dangerous_match = re.search(dangerous_pattern, text)
        if dangerous_match:
            try:
                dangerous = int(dangerous_match.group(1))
            except ValueError:
                pass

        # Extract reason
        reason = "Extracted from response"  # Default value
        reason_match = re.search(reason_pattern, text)
        if reason_match:
            reason = reason_match.group(1)

        return {"code": code, "dangerous": dangerous, "reason": reason}

    # Method 3: Try to fix and balance braces in malformed JSON
    try:
        start_idx = text.find("{")
        if start_idx >= 0:
            # Find the matching end brace or add it if missing
            json_text = text[start_idx:]

            # Count opening and closing braces
            open_count = json_text.count("{")
            close_count = json_text.count("}")

            # If unbalanced, add missing closing braces
            if open_count > close_count:
                json_text += "}" * (open_count - close_count)

            try:
                parsed_json = json.loads(json_text)
                if isinstance(parsed_json, dict) and "code" in parsed_json:
                    return parsed_json
            except json.JSONDecodeError:
                # Try with cleaned JSON
                cleaned_json = re.sub(r"[\t\n\r]", " ", json_text)
                try:
                    parsed_json = json.loads(cleaned_json)
                    if isinstance(parsed_json, dict) and "code" in parsed_json:
                        return parsed_json
                except json.JSONDecodeError:
                    pass
    except Exception:
        pass

    # Method 4: Extract code directly if all JSON parsing fails
    code = extract_code_from_markdown(text)
    if code:
        return {
            "code": code,
            "dangerous": 1,
            "reason": "Parser couldn't extract danger assessment, using default safe level.",
        }

    return None


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
        print(f"Standard parsing failed: {str(e)}")

        # Second try with the fixing parser
        try:
            print("Attempting to fix malformed output...")
            return fixing_parser.parse(response_text)
        except OutputParserException as e2:
            print(f"Fixing parser failed: {str(e2)}")

            # Third try to manually extract JSON
            print("Attempting manual JSON extraction...")
            json_data = extract_json_manually(response_text)

            if json_data:
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
                    print(f"Failed to create model from extracted JSON: {str(e3)}")

            # Final fallback: extract code and create a basic analysis
            code = extract_code_from_markdown(response_text)
            return CodeAnalysis(
                code=code,
                dangerous=1,
                reason="Parser couldn't extract danger assessment, using default safe level.",
            )


def extract_json_from_text(text):
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


def extract_code_from_markdown(text):
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
