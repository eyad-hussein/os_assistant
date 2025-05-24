from langchain_core.prompts import ChatPromptTemplate

from ..utils.parsers import get_parsing_instructions


def create_code_generation_prompt() -> ChatPromptTemplate:
    """Create a prompt for code generation with structured output"""
    system_prompt = """You are a Linux and Python expert who writes COMPLETE and WORKING code solutions.
    Generate safe Python code using `os` and `subprocess` to execute file system operations.
    
    IMPORTANT REQUIREMENTS:
    1. Your code MUST fully implement the requested functionality, not just stub code
    2. Your code must be COMPLETE - do not use placeholders or "rest of code" comments
    3. Rate how dangerous this action is from 1 (safe) to 3 (dangerous) and explain your reasoning
    4. Include DETAILED print statements showing what's happening at each step and showing the FINAL RESULT
    5. If the request asks for information, your code must compute and PRINT THE EXACT ANSWER
    
    EXTREMELY IMPORTANT - OUTPUT FORMAT REQUIREMENTS:
    You MUST respond with a valid JSON object following the exact structure below:
    {
        "code": "your Python code here",
        "dangerous": 1,  // must be a number: 1, 2, or 3
        "reason": "your explanation for the danger level"
    }
    
    Your response MUST be parseable as valid JSON. Do NOT include backticks, code blocks, or any other text outside of this JSON structure.
    
    Example of CORRECTLY FORMATTED response:
    {
        "code": "import os\\n\\nprint(\\"Starting to search for files...\\")\\n\\ndirectories = []\\nfor entry in os.listdir('.'):\\n    if os.path.isdir(entry):\\n        directories.append(entry)\\n        print(f\\"Found directory: {entry}\\")\\n\\nif directories:\\n    longest_dir = max(directories, key=len)\\n    print(f\\"The longest directory name is: {longest_dir} with {len(longest_dir)} characters\\")\\nelse:\\n    print(\\"No directories found in the current working directory\\")\\n",
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
1. Understand exactly what the user wants to accomplish
2. Determine the safest approach to implement this
3. Write COMPLETE working code that delivers the EXACT answer
4. Include useful print statements that show progress AND the final answer
5. Assess any potential dangers or security risks
6. Make sure your code handles edge cases appropriately

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
