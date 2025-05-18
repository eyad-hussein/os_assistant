from langchain_core.prompts import ChatPromptTemplate

from .parsers import get_parsing_instructions


def create_code_generation_prompt() -> ChatPromptTemplate:
    """Create a prompt for code generation with structured output"""
    system_prompt = """You are a Linux and Python expert. Generate safe Python code using `os` and `subprocess`
    to execute file system operations.
    Rate how dangerous this action is from 1 (safe) to 3 (dangerous) and explain your reasoning.
    Be precise and follow the output format exactly.
    """

    template = """
{system_prompt}

User request: {instruction}

{format_instructions}

Think through this step by step:
1. Understand what the user wants to accomplish
2. Determine the safest approach to implement this
3. Assess any potential dangers or security risks
4. Generate well-documented Python code
5. Provide a danger assessment with clear reasoning
"""

    return ChatPromptTemplate.from_template(
        template=template,
        partial_variables={
            "system_prompt": system_prompt,
            "format_instructions": get_parsing_instructions(),
        },
    )


def create_code_error_prompt() -> ChatPromptTemplate:
    """Create a prompt for handling code errors"""
    template = """You need to fix Python code that encountered an error:

Original question: {question}

Code executed:
```python
{code}
```

Error encountered:
{error}

Output so far:
{output}

Please fix the code. Add print statements to help debug the issue.

{format_instructions}
"""
    return ChatPromptTemplate.from_template(
        template=template,
        partial_variables={"format_instructions": get_parsing_instructions()},
    )


def create_summary_prompt() -> ChatPromptTemplate:
    """Create a prompt for summarizing execution results"""
    template = """You executed the following Python code:
```python
{code}
```

The code produced this output:
{stdout}

Please provide a clear summary of what the code did and what the results mean.
Explain any potential security or safety implications.
"""
    return ChatPromptTemplate.from_template(template)
