import re

from .string_utils import ensure_string


def extract_code_from_markdown(text: str, language: str | None = None) -> str:
    text = ensure_string(text)

    if language:
        # Match ```python ... ```
        pattern = rf"```{language}\s*(.*?)```"
    else:
        # Match ```...``` regardless of language
        pattern = r"```(?:\w+)?\s*(.*?)```"

    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return ""
