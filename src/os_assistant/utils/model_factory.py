from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

import os_assistant.utils.settings as settings


def create_model(**overrides):
    """Create a model instance based on the provider specified in settings."""
    model_type = overrides.get("model_type", settings.MODEL_TYPE)

    if model_type == "OPENAI":
        return ChatOpenAI(
            model=overrides.get("model", settings.MODEL_NAME),
            # temperature=overrides.get("temperature", settings.TEMPERATURE),
            max_tokens=overrides.get("max_tokens", None),
            timeout=overrides.get("timeout", None),
            max_retries=overrides.get("max_retries", 2),
            api_key=overrides.get("api_key", settings.OPENAI_API_KEY),
        )
    elif model_type == "ANTHROPIC":
        return ChatAnthropic(
            model=overrides.get("model", settings.MODEL_NAME),
            temperature=overrides.get("temperature", settings.TEMPERATURE),
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


# Primary model
model = create_model(model_type="OPENAI")

# Backup model for fixing outputs
fixing_model = create_model(model_type="OPENAI")

# Coding Model
coding_model = create_model(
    model_type="OPENAI",
    model=settings.LLM_MODEL_CODING,
)
