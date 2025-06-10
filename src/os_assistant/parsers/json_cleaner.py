import json
import re
from typing import Any, Dict, List, Union


def clean_and_parse_json(text: str) -> Union[Dict[str, Any], List[Any], None]:
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
    except json.JSONDecodeError as e:
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
                # As a last resort, try a more aggressive approach to extract valid JSON
                try:
                    # Find anything that looks like JSON
                    potential_json = re.search(
                        r"({[\s\S]*?}|\[[\s\S]*?\])", cleaned_text
                    )
                    if potential_json:
                        return json.loads(potential_json.group(0))
                    return None
                except:
                    return None
