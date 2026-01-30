import json
import re
from typing import Any


def clean_and_parse_json(text: str) -> dict[str, Any] | list[Any] | None:
    """
    Clean and parse JSON text that might contain PowerShell or other problematic syntax.

    Args:
        text: String containing JSON-like content that may need cleaning

    Returns:
        Parsed JSON object (dict or list) or None if parsing fails
    """
    if not text:
        return None

    # Make a copy of the original text
    cleaned_text = str(text)

    # Step 1: Handle PowerShell variables and syntax issues
    # Replace $_ with $item to avoid JSON parsing issues
    cleaned_text = cleaned_text.replace("$_", "$item")

    # Replace $. with $item. for object property access
    cleaned_text = cleaned_text.replace("$.", "$item.")

    # Escape PowerShell variables by doubling $ (but avoid doubling already doubled $)
    cleaned_text = re.sub(r"(?<!\$)\$([a-zA-Z_][a-zA-Z0-9_]*)", r"$$\1", cleaned_text)

    # Handle PowerShell hash tables
    cleaned_text = re.sub(r"@{([^}]*)}", r"{'\1'}", cleaned_text)

    # Handle PowerShell arrays
    cleaned_text = cleaned_text.replace("@(", "[").replace(")", "]")

    # Fix PowerShell parameter notation
    cleaned_text = re.sub(r"-([a-zA-Z]+)\s+", r"'-\1': ", cleaned_text)

    # Step 2: Fix common JSON formatting issues
    # Remove leading/trailing whitespace
    cleaned_text = cleaned_text.strip()

    # Ensure proper quotes for JSON keys and string values
    # Replace single quotes with double quotes for JSON keys
    cleaned_text = re.sub(r"'([^']+)'(\s*:)", r'"\1"\2', cleaned_text)

    # Step 3: Handle PowerShell command-specific issues
    # Fix pipeline operator spacing to ensure it's preserved
    cleaned_text = re.sub(r"\s*\|\s*", " | ", cleaned_text)

    # Preserve command switches and parameters
    cleaned_text = re.sub(r"-([a-zA-Z]+)", r"-\1", cleaned_text)

    # Step 4: Extract the JSON object if it's embedded in other text
    json_match = re.search(r"({[\s\S]*}|\[[\s\S]*\])", cleaned_text)
    if json_match:
        cleaned_text = json_match.group(0)

    # Step 5: Try to parse the JSON
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        # If initial parsing fails, try additional fixes
        try:
            # Handle trailing commas in arrays or objects
            cleaned_text = re.sub(r",\s*([\]}])", r"\1", cleaned_text)
            return json.loads(cleaned_text)
        except json.JSONDecodeError:
            # Handle missing quotes around property names
            try:
                cleaned_text = re.sub(
                    r"([{,]\s*)([a-zA-Z0-9_]+)(\s*:)", r'\1"\2"\3', cleaned_text
                )
                return json.loads(cleaned_text)
            except json.JSONDecodeError:
                # Handle PowerShell variable and string concatenation like $env:TEMP
                try:
                    # Replace $env:TEMP and similar with placeholders
                    cleaned_text = re.sub(
                        r"\$env:[A-Za-z_][A-Za-z0-9_]*", "ENV_VARIABLE", cleaned_text
                    )
                    return json.loads(cleaned_text)
                except json.JSONDecodeError:
                    # Handle PowerShell subexpressions $()
                    try:
                        cleaned_text = re.sub(
                            r"\$\(.*?\)", "SUBEXPRESSION", cleaned_text
                        )
                        return json.loads(cleaned_text)
                    except json.JSONDecodeError:
                        # As a last resort, try a more aggressive approach to extract valid JSON
                        try:
                            # Find anything that looks like JSON
                            potential_json = re.search(
                                r"({[\s\S]*?}|\[[\s\S]*?\])", cleaned_text
                            )
                            if potential_json:
                                return json.loads(potential_json.group(0))

                            # If we still can't parse it, try to build a dict manually
                            command_match = re.search(
                                r'"command"\s*:\s*"([^"]*)"', cleaned_text
                            )
                            explanation_match = re.search(
                                r'"explanation"\s*:\s*"([^"]*)"', cleaned_text
                            )

                            if command_match:
                                result = {"command": command_match.group(1)}
                                if explanation_match:
                                    result["explanation"] = explanation_match.group(1)
                                return result

                            # For information responses
                            answer_match = re.search(
                                r'"answer"\s*:\s*"([^"]*)"', cleaned_text
                            )
                            if answer_match:
                                return {"answer": answer_match.group(1)}

                            return None
                        except Exception:
                            return None


def extract_json_objects(text: str) -> list[dict[str, Any]]:
    """
    Extract all JSON-like objects from text by finding balanced braces.

    Args:
        text: String that may contain JSON objects

    Returns:
        List of parsed JSON objects found in the text
    """
    results = []

    # First, normalize the text - replace newlines with spaces for easier processing
    text = re.sub(r"\s+", " ", text)

    # Start positions of potential objects
    starts = [match.start() for match in re.finditer(r"{", text)]

    for start in starts:
        # Track nested braces
        brace_count = 0
        end = -1
        in_string = False
        escape_next = False

        for i in range(start, len(text)):
            char = text[i]

            # Handle string literals (avoid counting braces inside strings)
            if char == '"' and not escape_next:
                in_string = not in_string
            elif char == "\\" and in_string:
                escape_next = True
                continue

            if not in_string:
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end = i
                        break

            escape_next = False

        if end != -1:
            # Found a balanced object, try to parse it
            potential_json = text[start : end + 1]
            try:
                # Clean and parse the object
                cleaned = clean_and_parse_json(potential_json)
                if cleaned:
                    results.append(cleaned)
            except Exception:
                pass

    return results
