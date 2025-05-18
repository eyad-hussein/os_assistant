from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from .config import OLLAMA_BASE_URL, LLM_MODEL, LLM_TEMPERATURE
from .models import CodeExecutionState, CodeAnalysis
from .executors import execute_code_in_memory
from .prompts import create_code_generation_prompt, create_code_error_prompt, create_summary_prompt
from .parsers import parse_structured_output, extract_code_from_markdown

def create_llm():
    """Create and configure the LLM"""
    return ChatOllama(
        model=LLM_MODEL, 
        temperature=LLM_TEMPERATURE,
        base_url=OLLAMA_BASE_URL
    )

def code_executor_agent(state: CodeExecutionState) -> CodeExecutionState:
    """
    A node that executes code and updates the state.
    """
    llm = create_llm()
    
    # If we have code in the state already, it means we're in a loop
    if state.code:
        # Execute the current code
        code_result = execute_code_in_memory(
            state.code, 
            danger_analysis=state.danger_analysis
        )
        
        # Update state with execution results
        state.execution_result = code_result["stdout"] if code_result["stdout"] else "No output"
        state.error_code = code_result["stderr"]
        
        # If there's an error, update question to include error info and return to try again
        if code_result["stderr"]:
            print(f"Encountered error: {code_result['stderr']}")
            print("Asking LLM to fix the error...")
            
            error_prompt = create_code_error_prompt()
            error_response = llm.invoke(
                error_prompt.format(
                    question=state.question,
                    code=state.code,
                    error=code_result["stderr"],
                    output=code_result["stdout"] if code_result["stdout"] else "No output"
                )
            )
            
            try:
                parsed_result = parse_structured_output(error_response, CodeAnalysis)
                state.code = parsed_result.code
                state.danger_analysis = {
                    "level": parsed_result.dangerous,
                    "reason": parsed_result.reason
                }
                print(f"Generated fixed code with danger level: {parsed_result.dangerous}")
            except Exception as e:
                print(f"Error parsing LLM response: {str(e)}")
                # Fallback to simple code extraction if parsing fails
                state.code = extract_code_from_markdown(error_response)
        else:
            # No errors, generate summary
            summary_prompt = create_summary_prompt()
            
            summary_response = llm.invoke(
                summary_prompt.format(
                    code=state.code, 
                    stdout=code_result["stdout"] if code_result["stdout"] else "No output"
                )
            )
            
            state.agent_output = summary_response
    else:
        # Initial execution - generate and execute code
        generation_prompt = create_code_generation_prompt()
        
        response = llm.invoke(
            generation_prompt.format(instruction=state.question)
        )
        
        try:
            # Parse the structured output
            parsed_result = parse_structured_output(response, CodeAnalysis)
            generated_code = parsed_result.code
            
            # Store danger analysis
            state.danger_analysis = {
                "level": parsed_result.dangerous,
                "reason": parsed_result.reason
            }
        except Exception:
            # Fallback to basic code extraction if parsing fails
            generated_code = extract_code_from_markdown(response)
        
        # Store the generated code
        state.code = generated_code
        
        # Execute the generated code
        code_result = execute_code_in_memory(
            generated_code,
            danger_analysis=state.danger_analysis
        )
        
        # Update state with execution results
        state.execution_result = code_result["stdout"] if code_result["stdout"] else "No output"
        state.error_code = code_result["stderr"]
        
        # If there's an error, prepare to rerun
        if code_result["stderr"]:
            error_prompt = create_code_error_prompt()
            error_response = llm.invoke(
                error_prompt.format(
                    question=state.question,
                    code=state.code,
                    error=code_result["stderr"],
                    output=code_result["stdout"] if code_result["stdout"] else "No output"
                )
            )
            
            try:
                parsed_result = parse_structured_output(error_response, CodeAnalysis)
                state.code = parsed_result.code
                state.danger_analysis = {
                    "level": parsed_result.dangerous,
                    "reason": parsed_result.reason
                }
            except Exception:
                # Fallback to simple code extraction if parsing fails
                state.code = extract_code_from_markdown(error_response)
        else:
            # No errors, generate summary
            summary_prompt = create_summary_prompt()
            
            summary_response = llm.invoke(
                summary_prompt.format(
                    code=generated_code, 
                    stdout=code_result["stdout"] if code_result["stdout"] else "No output"
                )
            )
            
            state.agent_output = summary_response
    
    return state

def router(state: CodeExecutionState):
    """Determine next node based on state"""
    # If there's an error and no final output, we need to loop back
    if state.error_code and not state.agent_output:
        return "code_executor"
    # Otherwise, we're done
    return END

def create_code_execution_graph():
    """Create and configure the execution graph"""
    workflow = StateGraph(CodeExecutionState)
    workflow.add_node("code_executor", code_executor_agent)

    # Connect the nodes with conditional routing
    workflow.add_edge(START, "code_executor")
    workflow.add_conditional_edges("code_executor", router)

    # Compile the graph
    return workflow.compile()
