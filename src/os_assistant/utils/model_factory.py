from langchain_ollama import ChatOllama
import os_assistant.utils.settings as settings

def create_model(**overrides):
    return ChatOllama(
        model=overrides.get("model", settings.MODEL_NAME),
        temperature=overrides.get("temperature", settings.TEMPERATURE),
        base_url=overrides.get("base_url", settings.MODEL_BASE_URL),
    )

# Primary model
model = create_model()

# Backup model for fixing outputs
fixing_model = create_model()
