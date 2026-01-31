from typing import Any

from langchain_core.messages import AIMessage


def ensure_string(message: Any) -> str:
    """Convert any message object to a string."""
    if isinstance(message, AIMessage):
        return str(message.content)
    elif hasattr(message, "content"):
        return str(message.content)
    return str(message)
