from os_assistant.utils.settings import ASSISTANT_MODE

def is_rag_enabled() -> bool:
    return ASSISTANT_MODE in [0, 2]

def is_code_execution_enabled() -> bool:
    return ASSISTANT_MODE in [0, 1]