from os_assistant.core.nodes.helpers import is_code_execution_enabled
from os_assistant.core.state import AssistantState
from os_assistant.tools.code_agent.wrapper import code_execute_tool


def tool_execution_node(state: AssistantState) -> AssistantState:
    """Execute a tool and store the results in the state"""
    print("\nNODE: tool_execution_node")

    # Check if code tool is enabled
    if not is_code_execution_enabled():
        print(
            "WARNING: Tool execution node called but code tool is disabled in current mode."
        )
        state["tool_context"] = (
            "The code execution tool is disabled in the current mode."
        )
        return state

    # Extract the question from the state
    question = str(state.get("tool_question", ""))
    if not question:
        print("Error: No tool question found in state.")
        state["tool_context"] = (
            "Error: No question was provided for the tool to execute."
        )
        return state

    print(f"Tool question: {question}")

    try:
        # Execute the question
        tool_state = code_execute_tool(question)
        # Check if execution was aborted due to too many errors
        if "Too many consecutive errors" in (tool_state.get("error_code") or ""):
            print("Tool execution aborted: Too many consecutive errors")

            # Create an error message to include in the state
            error_message = f"""
            I attempted to execute code to answer your question, but encountered multiple errors.
            
            Question: {question}
            
            After 3 failed attempts, I had to abort execution for safety reasons.
            Please try simplifying your request or provide more specific instructions.
            """

            state["tool_context"] = error_message
            return state

        print("Tool execution completed successfully.")
        print(f"Code executed: {tool_state['code']}")
        print(
            f"Execution result: {tool_state['execution_result'][:100]}..."
            if len(tool_state["execution_result"]) > 100
            else f"Execution result: {tool_state['execution_result']}"
        )

        # Prepare a message to add to the state that will be used when returning to the originating node
        tool_context = f"""
        I used the code_execute_tool to answer your question.
        
        Question: {question}
        
        Code used: {tool_state["code"]}
        
        ===== RAW EXECUTION RESULTS (DO NOT MODIFY THESE) =====
        {tool_state["execution_result"]}
        ===== END OF RAW RESULTS =====
        
        Analysis: {tool_state["agent_output"]}
        
        IMPORTANT: You MUST include the complete raw execution results above in your response, exactly as shown. Do not summarize, truncate, or modify them in any way. The user needs to see the exact, unedited output from the system.
        """

        state["tool_context"] = tool_context
        # Store raw results for later use in the new fields
        state["raw_tool_results"] = tool_state["execution_result"]
        state["tool_code"] = tool_state["code"]
        state["tool_analysis"] = tool_state["agent_output"]

    except Exception as e:
        print(f"Error executing tool: {str(e)}")
        state["tool_context"] = (
            f"An error occurred while executing the tool: {str(e)}\n\nThis might be due to system limitations or the complexity of the request. Please try a simpler question or provide more specific details."
        )

    print("EXITING tool_execution_node")
    print(f"Modified state keys: {state.keys()}")
    print(f"Prompt value: {state.get('prompt')}")
    return state
