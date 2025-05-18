from langchain_core.tools import tool
from .main import run_code_execution
@tool
def code_execute_tool(question: str) -> dict:
   """Tool to execute code based on a user's question.
   Args:
       question (str): The question or code to execute."""
   tool_state = run_code_execution(question, verbose=True)

   return {
         "question": question,
         "code": tool_state["code"],
         "danger_analysis": tool_state["danger_analysis"],
         "execution_result": tool_state["execution_result"],
         "error_code": tool_state["error_code"],
         "agent_output": tool_state["agent_output"]
   }