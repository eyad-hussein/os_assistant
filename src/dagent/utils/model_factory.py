from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

import dagent.utils.settings as settings


def create_model(**overrides):
    """Create a model instance based on the provider specified in settings."""
    model_type = overrides.get("model_type", settings.MODEL_TYPE)

    if model_type == "OLLAMA":
        return ChatOllama(
            model=overrides.get("model", settings.MODEL_NAME),
            temperature=overrides.get("temperature", settings.TEMPERATURE),
            base_url=overrides.get("base_url", settings.MODEL_BASE_URL),
        )
    elif model_type == "OPENAI":
        return ChatOpenAI(
            model=overrides.get("model", settings.MODEL_NAME),
            temperature=overrides.get("temperature", settings.TEMPERATURE),
            max_tokens=overrides.get("max_tokens", None),
            timeout=overrides.get("timeout", None),
            max_retries=overrides.get("max_retries", 2),
        )
    elif model_type == "ANTHROPIC":
        return ChatAnthropic(
            model=overrides.get("model", settings.MODEL_NAME),
            temperature=overrides.get("temperature", settings.TEMPERATURE),
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


# Primary model
model = create_model()

# Backup model for fixing outputs
fixing_model = create_model()

# Coding Model
coding_model = create_model(
    model=settings.LLM_MODEL_CODING,
)
