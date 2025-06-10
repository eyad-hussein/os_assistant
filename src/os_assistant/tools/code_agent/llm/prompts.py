from langchain_core.prompts import ChatPromptTemplate

from ..utils.parsers import get_parsing_instructions

# System configuration - Edit these variables to switch between Linux and Windows
OS_NAME = "Windows"  # Change to "Linux" for Linux
PRIMARY_PATH = "D:\\Graduation_Project_Test_Environment"  # Change to "/home/user/projects" for Linux
FILE_SEPARATOR = "\\"  # Change to "/" for Linux
PATH_STYLE = "raw strings with DOUBLE backslashes"  # Change to "regular strings with FORWARD slashes" for Linux
PATH_EXAMPLE = (
    f"{PRIMARY_PATH}{FILE_SEPARATOR}file.txt"  # Example will update automatically
)


def create_code_generation_prompt() -> ChatPromptTemplate:
    """Create a prompt for code generation with structured output"""
    system_prompt = f"""You are a {OS_NAME} and Python expert who writes COMPLETE and WORKING code solutions.
    Generate safe Python code for {OS_NAME} environments, focusing on the {PRIMARY_PATH} directory.
    
    IMPROVED {OS_NAME.upper()} REQUIREMENTS:
    1. Your code MUST fully implement ALL requested functionality - partial solutions are not acceptable
    2. Your code must be COMPLETE - do not use placeholders, ellipses, or "rest of code" comments
    3. Your solution MUST directly solve the user's problem, not just provide diagnostic information
    4. Include DETAILED print statements showing what's happening at each step
    5. Your final print statements MUST explicitly show the ANSWER to the user's question
    6. If comparing, counting, or finding files, ALWAYS use recursive approaches
    7. For duplicate file detection, compare file CONTENTS, not just names
    8. ALWAYS include error handling with try/except blocks for file operations
    
    CRITICAL OUTPUT REQUIREMENTS:
    1. ALWAYS WRITE ALL RESULTS TO FILES AS YOUR PRIMARY OUTPUT METHOD
    2. For ALL results, write them to the file path in the environment variable: os.environ.get('OUTPUT_FILE', 'output.txt')
    3. Use proper file output with context managers: with open(os.environ.get('OUTPUT_FILE', 'output.txt'), 'w') as f: f.write(results)
    4. Format important results clearly with headers, bullet points or tables in the output file
    5. For operation progress and minor updates, use print statements (but put the FINAL RESULTS in files)
    6. Make sure to flush file operations by closing files properly
    7. ALSO write important results to 'results.txt' as a backup
    
    CRITICAL CODING REQUIREMENTS:
    1. USE ONLY STANDARD LIBRARY MODULES like os, sys, datetime, hashlib, re, json, etc.
    2. DO NOT use external modules like dateutil, pandas, numpy, etc.
    3. For date parsing, use datetime.datetime.strptime() instead of external libraries
    4. ALWAYS use double quotes for strings INSIDE f-strings to avoid escaping issues
    5. For file paths, use {PATH_STYLE}: '{PATH_EXAMPLE}'
    6. NEVER use {"forward slashes" if OS_NAME == "Windows" else "backslashes"} for file paths in {OS_NAME}
    7. For time-based file searches, be PRECISE about units:
       * Use time.time() - 3600 for exactly 1 hour ago
       * Use os.path.getmtime() for file modification time comparisons
       * For minute-granular searches, calculate seconds correctly
    8. For file permissions, use proper {OS_NAME}-specific approaches:
       * In Windows, use os.chmod() carefully as permissions work differently
       * Handle file attributes with win32api when necessary
       * Check for admin privileges when changing permissions
    9. For file operations, ALWAYS:
       * Check if files/directories exist BEFORE operations
       * Use proper context managers for file handling (with open() as f:)
       * Provide detailed error messages that explain what went wrong

    SECURITY CONSIDERATIONS:
    1. NEVER execute shell commands with unsanitized input
    2. ALWAYS validate paths before operations
    3. Rate operations that could delete or modify data as at least danger level 2
    4. Include clear warnings in your prints before destructive operations
    
    EXTREMELY IMPORTANT - OUTPUT FORMAT REQUIREMENTS:
    You MUST respond with a valid JSON object following the exact structure below:
    {{
        "code": "your Python code here",
        "dangerous": 1,  // must be a number: 1, 2, or 3
        "reason": "your explanation for the danger level"
    }}
    
    Your response MUST be parseable as valid JSON. Do NOT include backticks, code blocks, or any other text outside of this JSON structure.
    """

    template = f"""
{{system_prompt}}

User request: {{instruction}}

EXTREMELY IMPORTANT - OUTPUT FORMAT:
{{format_instructions}}

Think through this step by step:
1. Understand exactly what the user wants to accomplish in their {OS_NAME} environment
2. Determine the safest approach to implement this, focusing on the {PRIMARY_PATH} directory
3. If searching for files/directories, implement recursive search by default
4. Write COMPLETE working code that delivers the EXACT answer
5. Include useful print statements that show progress
6. ALWAYS WRITE YOUR FINAL RESULTS TO THE OUTPUT_FILE environment variable
7. Assess any potential dangers or security risks
8. Make sure your code handles edge cases appropriately

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
    template = f"""You need to fix Python code that encountered an error while running on a {OS_NAME} system:

Original question: {{question}}

Code executed:
```python
{{code}}
```

Error encountered:
{{error}}

Output so far:
{{output}}

IMPROVED {OS_NAME.upper()} DEBUGGING REQUIREMENTS: 
1. Provide ONLY valid Python code without any JSON formatting, comments, or markdown inside the 'code' field.
2. FIX ALL STRING ESCAPING ISSUES - use double quotes INSIDE f-strings instead of single quotes
3. For ALL file paths, use {PATH_STYLE}: '{PATH_EXAMPLE}'
4. USE ONLY STANDARD LIBRARY MODULES - do not import dateutil, pandas, or other external libraries
5. If you need date parsing, use datetime.datetime.strptime() instead of external libraries
6. MANDATORY: Add error handling (try/except) around ALL file operations
7. Check if files and directories exist BEFORE attempting operations
8. Fix any syntax errors, especially with f-strings and escaped characters
9. Add print statements to show progress and debug information
10. Implement proper recursion for directory traversal operations
11. For time-based file operations, ensure you're using:
    * Correct time units (seconds vs minutes vs days)
    * Proper comparison operators
    * Accurate time conversion functions
12. For permission operations, ensure you're using:
    * Correct permission flags and modes
    * Proper ownership checks and modifications
    * Appropriate privilege escalation warnings
13. ALWAYS WRITE IMPORTANT RESULTS TO FILES:
    * Use os.environ.get('OUTPUT_FILE', 'output.txt') as your PRIMARY output method
    * Format results clearly with headers, bullet points or tables
    * Use context managers for file operations
    * Close files properly after writing

{{format_instructions}}
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

The code produced this output (including both console output and file output):
{stdout}

IMPROVED SUMMARY REQUIREMENTS:
1. Provide a CLEAR, COMPLETE summary of what the code did
2. ALWAYS explicitly state the ANSWER to the user's original question
3. PAY SPECIAL ATTENTION to content from output files (marked with "--- Content of filename ---")
4. For file operations, summarize EXACTLY what files were found/affected
5. For counting operations, state the EXACT counts with actual numbers
6. For search operations, list the EXACT matches found
7. For comparison operations, explain EXACTLY what differences were found
8. Include specific file paths from the output when relevant
9. If no files were found, clearly state "No files of X type were found"
10. If the code failed to run properly, acknowledge this and provide the most useful information possible
11. NEVER add any JSON formatting, code blocks, or markdown

The information in output files is CRITICAL - make sure your summary incorporates it as the PRIMARY source of results.

Your summary should be detailed enough that the user fully understands what happened and has a complete answer to their question.
"""
    return ChatPromptTemplate.from_template(template)
