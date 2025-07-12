from .string_utils import ensure_string


def extract_code_from_markdown(text: str) -> str:
    """Extract code from markdown code blocks."""
    text = ensure_string(text)

    if "```python" in text:
        code_blocks = text.split("```python")[1:]
        for block in code_blocks:
            if "```" in block:
                return block.split("```")[0].strip()

    elif "```" in text:
        code_blocks = text.split("```")[1::2]
        if code_blocks:
            return code_blocks[0].strip()

    return ""
