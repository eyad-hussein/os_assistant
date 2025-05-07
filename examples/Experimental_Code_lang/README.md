## Folder Structure

```
Experimental_Code_lang/
│
├── main.py                     # Entry point and main loop
│
├── src/                        # Source code directory
│   ├── config/                 # Configuration settings
│   │   └── settings.py         # LLM and domain configurations
│   │
│   ├── graph/                  # LangGraph components
│   │   ├── builder.py          # Graph construction and compilation
│   │   ├── edges.py            # Edge routing logic
│   │   ├── nodes.py            # Node processing functions
│   │   └── state.py            # State definition
│   │
│   ├── models/                 # Data models
│   │   └── schemas.py          # Pydantic models for structure validation
│   │
│   └── parsers/                # Output parsing
│       └── setup.py            # JSON parsing, validation, and error handling
│
└── domain_logs/                # Generated logs directory
```

## File Descriptions

- **main.py**: Application entry point, handles user input and graph execution
- **settings.py**: Configuration for the LLM model, domains, and logging
- **schemas.py**: Pydantic models that define the structure of input/output data
- **state.py**: TypedDict definition for the graph state including history
- **nodes.py**: Processing functions for each step in the workflow
- **edges.py**: Conditional routing logic between nodes
- **builder.py**: Constructs and compiles the LangGraph with nodes and edges
- **setup.py**: Parsers and utilities for handling LLM output, especially JSON

## Running the Application

Run `python main.py` from the root directory
