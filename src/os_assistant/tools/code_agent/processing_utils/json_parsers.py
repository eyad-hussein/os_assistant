import json
import re

from os_assistant.utils import LOGGER

from .string_utils import ensure_string


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
        LOGGER.info(f"Manual JSON extraction failed: {str(e)}")

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
