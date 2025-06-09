from langchain_core.prompts import ChatPromptTemplate

from ..utils.parsers import get_parsing_instructions


def create_code_generation_prompt() -> ChatPromptTemplate:
    """Create a prompt for code generation with structured output"""
    system_prompt = """You are a Windows 10 and Python expert who writes COMPLETE and WORKING code solutions.
    Generate safe Python code for Windows environments, focusing on the D:\Graduation_Project_Test_Environment directory.
    
    IMPROVED WINDOWS REQUIREMENTS:
    1. Your code MUST fully implement ALL requested functionality - partial solutions are not acceptable
    2. Your code must be COMPLETE - do not use placeholders, ellipses, or "rest of code" comments
    3. Your solution MUST directly solve the user's problem, not just provide diagnostic information
    4. Include DETAILED print statements showing what's happening at each step
    5. Your final print statements MUST explicitly show the ANSWER to the user's question
    6. If comparing, counting, or finding files, ALWAYS use recursive approaches
    7. For duplicate file detection, compare file CONTENTS, not just names
    8. ALWAYS include error handling with try/except blocks for file operations
    
    FILE OPERATION SPECIFICS:
    1. Use Windows-style paths with BACKSLASHES in raw strings: r'D:\\Graduation_Project_Test_Environment'
    2. For recursive file operations, use os.walk() or pathlib's rglob()
    3. When reading files, use appropriate encoding parameters
    4. Always include content comparison for duplicate detection (via hashing)
    5. For file searching, implement full recursive directory traversal
    
    CRITICAL IMPLEMENTATION PATTERNS:
    1. For finding text in files: Open each file and check contents, don't just check names
    2. For finding duplicates: Generate and compare file hashes (use hashlib)
    3. For counting files by type: Use recursive search and dictionary counters
    4. For comparing directories: Implement detailed comparison with sets or dictionaries
    5. For permission operations: Use appropriate Windows functions (not chmod/chown)
    
    SECURITY CONSIDERATIONS:
    1. NEVER execute shell commands with unsanitized input
    2. ALWAYS validate paths before operations
    3. Rate operations that could delete or modify data as at least danger level 2
    4. Include clear warnings in your prints before destructive operations
    
    EXTREMELY IMPORTANT - OUTPUT FORMAT REQUIREMENTS:
    You MUST respond with a valid JSON object following the exact structure below:
    {
        "code": "your Python code here",
        "dangerous": 1,  // must be a number: 1, 2, or 3
        "reason": "your explanation for the danger level"
    }
    
    Your response MUST be parseable as valid JSON. Do NOT include backticks, code blocks, or any other text outside of this JSON structure.
    """

    template = """
{system_prompt}

User request: {instruction}

EXTREMELY IMPORTANT - OUTPUT FORMAT:
{format_instructions}

Think through this step by step:
1. Understand exactly what the user wants to accomplish in their Windows environment
2. Determine the safest approach to implement this, focusing on the D:\Graduation_Project_Test_Environment directory
3. If searching for files/directories, implement recursive search by default
4. Write COMPLETE working code that delivers the EXACT answer
5. Include useful print statements that show progress AND the final answer
6. Assess any potential dangers or security risks
7. Make sure your code handles edge cases appropriately

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
    template = """You need to fix Python code that encountered an error while running on a Windows 10 system:

Original question: {question}

Code executed:
```python
{code}
```

Error encountered:
{error}

Output so far:
{output}

IMPROVED WINDOWS DEBUGGING REQUIREMENTS: 
1. Provide ONLY valid Python code without any JSON formatting, comments, or markdown inside the 'code' field.
2. MANDATORY: Your fixed code MUST INCLUDE AT LEAST 5-10 PRINT STATEMENTS.
   Add print statements before and after each operation to explain what's happening.
   Make sure to print variable values, especially those involved in the error.
3. ALWAYS use raw strings (r'path') for Windows file paths: r'D:\\folder'
4. Ensure all paths use BACKSLASHES not forward slashes
5. Add robust error handling (try/except) around ALL file operations
6. Check if files and directories exist BEFORE attempting operations
7. Fix any issues with string formatting or variable references
8. Ensure all imports are at the top of the file
9. If working with files, handle encodings properly with 'encoding="utf-8"'
10. Implement proper recursion for directory traversal operations

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

IMPROVED SUMMARY REQUIREMENTS:
1. Provide a CLEAR, COMPLETE summary of what the code did
2. ALWAYS explicitly state the ANSWER to the user's original question
3. For file operations, summarize EXACTLY what files were found/affected
4. For counting operations, state the EXACT counts
5. For search operations, list the EXACT matches found
6. For comparison operations, explain EXACTLY what differences were found
7. Include specific file paths from the output when relevant
8. Explain any security implications of the operations performed
9. Format your response with clear sections and bullet points
10. NEVER add any JSON formatting, code blocks, or markdown

Your summary should be detailed enough that the user fully understands what happened and has a complete answer to their question.
"""
    return ChatPromptTemplate.from_template(template)
