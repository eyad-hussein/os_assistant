# OS Assistant Code Agent

A tool that allows an AI assistant to generate and execute Python code safely based on user questions or instructions.

## Files Overview

### Core Components

- **`core/models.py`**: Defines the data models used throughout the system.
  - `CodeAnalysis`: Pydantic model for code safety analysis
  - `CodeExecutionState`: State management for the code execution process

- **`core/run_code.py`**: Main entry point that orchestrates the code execution workflow.
  - Configures and runs the execution graph
  - Handles error reporting and state management

### LLM Integration

- **`llm/agents.py`**: Implements the state machine logic for code generation and execution.
  - Creates the LLM connection
  - Defines the code executor agent that generates, executes, and fixes code
  - Implements the routing logic and creates the execution graph

- **`llm/prompts.py`**: Contains the prompt templates for various code agent operations.
  - Code generation prompt with safety analysis
  - Error handling prompt
  - Summary generation prompt

### Execution

- **`execution/executors.py`**: Contains the code execution environment.
  - `execute_code_in_memory`: Primary method for executing code within the current process
  - `execute_code_in_subprocess`: Alternative method for more isolated execution
  - Implements safety checks and interactive confirmation for dangerous operations

### Utilities

- **`utils/parsers.py`**: Provides utilities for parsing and handling LLM outputs.
  - Creates and manages output parsers for structured responses
  - Extracts code from various formats (JSON, Markdown)
  - Handles conversion between different message types

### Entry Points

- **`wrapper.py`**: Provides a LangChain-compatible tool interface for the code agent.
  - Creates a tool that can be used in LangChain agents
  - Formats and standardizes the tool outputs

- **`main.py`**: Command line interface for standalone usage.
  - Parses command line arguments
  - Allows direct invocation of the code agent

