from .agents import create_code_execution_graph
from .models import CodeExecutionState
import argparse
import json
import os

def run_code_execution(question: str, verbose: bool = False, interactive: bool = True):
    """Run the code execution graph with the given question"""
    # Set interactive mode in environment for executors to access
    os.environ["INTERACTIVE_MODE"] = "1" if interactive else "0"
    
    # Initialize the graph
    code_execution_graph = create_code_execution_graph()
    
    # Set up the initial state
    initial_state = {"question": question}
    
    # Run the graph
    print(f"Processing question: {question}")
    print("="*50)
    
    final_state = code_execution_graph.invoke(initial_state)
    
    # Print results
    if verbose:
        print("\nFull execution details:")
        print("-"*50)
        print(f"Original question: {final_state['question']}")
        print(f"\nGenerated code:")
        print(f"```python\n{final_state['code']}\n```")
        
        if final_state['danger_analysis']:
            print(f"\nSafety analysis:")
            print(f"Danger level: {final_state['danger_analysis']['level']}/3")
            print(f"Reason: {final_state['danger_analysis']['reason']}")
        
        print(f"\nExecution output:")
        print(final_state['execution_result'] or "No output")
        
        if final_state['error_code']:
            print(f"\nErrors encountered:")
            print(final_state['error_code'])
    
    print("\nFinal summary:")
    print("-"*50)
    print(final_state['agent_output'])
    
    return final_state

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code execution agent")
    parser.add_argument("question", type=str, help="The question or code to execute")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--non-interactive", action="store_true", 
                        help="Run in non-interactive mode (no prompts for dangerous operations)")
    
    args = parser.parse_args()
    run_code_execution(args.question, args.verbose, not args.non_interactive)
