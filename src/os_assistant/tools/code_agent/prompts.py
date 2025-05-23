from langchain_core.prompts import ChatPromptTemplate

from .parsers import get_parsing_instructions


def create_code_generation_prompt() -> ChatPromptTemplate:
    """Create a prompt for code generation with structured output"""
    system_prompt = """You are a Linux and Python expert. Generate safe Python code using `os` and `subprocess`
    to execute file system operations.
    Rate how dangerous this action is from 1 (safe) to 3 (dangerous) and explain your reasoning.
    
    EXTREMELY IMPORTANT - OUTPUT FORMAT REQUIREMENTS:
    You MUST respond with a valid JSON object following the exact structure below:
    {
        "code": "your Python code here",
        "dangerous": 1,  // must be a number: 1, 2, or 3
        "reason": "your explanation for the danger level"
    }
    
    Your response MUST be parseable as valid JSON. Do NOT include backticks, code blocks, or any other text outside of this JSON structure.
    
    PRINT STATEMENT REQUIREMENTS:
    Your code MUST INCLUDE AT LEAST 5-10 PRINT STATEMENTS showing what's happening at each step.
    
    Example of CORRECTLY FORMATTED response:
    {
        "code": "import os\\n\\nprint(\\"Starting to search for files...\\")\\n# rest of code with many print statements",
        "dangerous": 1,
        "reason": "This code only reads files without modifying anything"
    }
    """

    template = """
{system_prompt}

User request: {instruction}

EXTREMELY IMPORTANT - OUTPUT FORMAT:
{format_instructions}

Think through this step by step:
1. Understand what the user wants to accomplish
2. Determine the safest approach to implement this
3. Assess any potential dangers or security risks
4. Generate well-documented Python code WITH AT LEAST 5-10 DETAILED PRINT STATEMENTS
5. Provide a danger assessment with clear reasoning

Remember: Your response MUST be a VALID JSON object that exactly matches the format specified.
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

IMPORTANT: 
1. Provide ONLY valid Python code without any JSON formatting, comments, or markdown inside the 'code' field.
2. MANDATORY: Your fixed code MUST INCLUDE AT LEAST 5-10 PRINT STATEMENTS.
   Add print statements before and after each operation to explain what's happening.
   Make sure to print variable values, especially those involved in the error.

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

IMPORTANT: Provide ONLY valid Python code without any JSON formatting, comments, or markdown inside the 'code' field.
"""
    return ChatPromptTemplate.from_template(template)
