@echo off
setlocal enabledelayedexpansion

:: Read each line from .env
for /f "usebackq tokens=1,2 delims==" %%A in (".env") do (
    set "VAR=%%A"
    set "VAL=%%B"

    :: Remove surrounding quotes if any
    set "VAL=!VAL:"=!"

    :: Dynamically set the environment variables
    set "!VAR!=!VAL!"
)

:: Set OLLAMA_HOST = MODEL_BASE_URL
set OLLAMA_HOST=%MODEL_BASE_URL%

:: Show what we're about to do
echo OLLAMA_HOST is set to %OLLAMA_HOST%

:: Pull models
ollama pull %MODEL_NAME%
ollama pull %CODING_AGENT_MODEL_NAME%
ollama pull %EMBEDDING_MODEL%

endlocal
