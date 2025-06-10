## Dev Notes
- Only `ChatOllama` models, such as `mistral:instruct`, are allowed in this version.
- This assistant is configured as a Windows expert by default.

## Prompt Optimization Notes

The prompts have been optimized based on evaluation results to address common issues:

### Key Improvements
1. **PowerShell vs Python Selection**: Clear guidelines on when to use each
2. **JSON Parsing Fixes**: Escape sequences for PowerShell variables to prevent JSON errors 
3. **Complex Task Handling**: Python scripts for complex file operations, PowerShell for simple ones
4. **Direct answers**: Explicit answers to "Is it possible..." questions
5. **Multiple approaches**: Alternative solutions for different Windows environments
6. **Better error handling**: Fallback mechanisms when commands fail

### Common Windows Command Problems Addressed
- **PowerShell variable syntax**: Using `$$_` instead of `$_` in JSON strings
- **Path handling**: Using raw strings like `r"D:\path"` in Python scripts
- **Complex operations**: Using Python for counting lines, file analysis, etc.
- **Script vs. Command**: Better decision logic on when to use each
- **Error handling**: Better exception management for file operations

## Quick OS Switching Guide

To switch between Windows and Linux:

### 1. Update Core Configuration

In `src/os_assistant/tools/code_agent/llm/prompts.py`, modify these variables at the top:
```python
# For Windows
OS_NAME = "Windows"
PRIMARY_PATH = "D:\\Graduation_Project_Test_Environment"
FILE_SEPARATOR = "\\"
PATH_STYLE = "raw strings with DOUBLE backslashes"

# For Linux
OS_NAME = "Linux"
PRIMARY_PATH = "/home/user/projects"
FILE_SEPARATOR = "/"
PATH_STYLE = "regular strings with FORWARD slashes"
```

### 2. Update Template Variables in YAML Files

For each YAML file in `src/os_assistant/prompts/` and `src/evaluator/prompts/`, replace these terms:

| Windows | Linux |
|---------|-------|
| Windows | Linux |
| D:\Graduation_Project_Test_Environment | /home/user/projects |
| folder | directory |
| PowerShell/CMD | shell script |
| Command Prompt | terminal |
| file system structure | filesystem hierarchy |
| PowerShell/CMD | bash/shell |
| backslashes | forward slashes |

### 3. Update Domain-Specific Commands

In `domain_analysis_node.yaml`:

| Domain | Windows Commands | Linux Commands |
|--------|-----------------|---------------|
| file_system | dir, copy, move, del, Get-ChildItem, icacls | ls, cp, mv, rm, find, chmod |


### 4. Swap Command Perspectives

In `command_generator_node.yaml` and `information_generator_node.yaml`:
- Windows → Include Linux equivalent notes
- Linux → Include Windows equivalent notes

