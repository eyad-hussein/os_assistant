import os
import sys
import uuid

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from src.config.settings import DOMAINS
from src.graph.builder import build_linux_assistant_graph
from src.graph.state import LinuxAssistantState
from src.utils.graph_visualizer import mermaid_to_png


def main():
    """Main function to run the Linux Assistant"""
    print("Building Linux Assistant graph...")
    app = build_linux_assistant_graph()

    try:
        graph = app.get_graph()
        mermaid_txt = graph.draw_mermaid()
        png_path = mermaid_to_png(mermaid_txt)
        print("Generated", png_path)
        print("Graph visualization saved to linux_assistant_graph.png")
    except Exception as e:
        print(f"Note: Graph visualization could not be generated: {e}")
        print("This is non-critical and the assistant will still function correctly.")

    print("Graph built successfully. Type 'exit' to quit.")

    # Create a persistent thread ID for this session
    # This is the key to short-term memory across invocations
    session_thread_id = str(uuid.uuid4())
    print(f"Session ID: {session_thread_id}")

    # Config dictionary for graph invocation
    config = {"configurable": {"thread_id": session_thread_id}}

    # Track the number of interactions to manage conversation length
    interaction_count = 0

    while True:
        try:
            user_prompt = input("\nEnter your Linux query: ")
            if user_prompt.lower() == "exit":
                print("Exiting Linux Assistant.")
                break
            if not user_prompt:
                continue

            interaction_count += 1

            # For the first interaction, we need to initialize the state
            if interaction_count == 1:
                # Initialize the state dictionary directly
                initial_state: LinuxAssistantState = {
                    "prompt": user_prompt,
                    "domains": DOMAINS,
                    "domain_analysis": None,
                    "contexts": {},
                    "domains_to_process": [],
                    "current_domain": None,
                    "query_type": None,
                    "command_response": None,
                    "information_response": None,
                    "final_result": None,
                    "conversation_history": [],
                    "conversation_summary": None,
                }

                # Invoke the graph with the initial state and thread config
                app.invoke(initial_state, config=config)
            else:
                # Retrieve the current state and update only the prompt
                current_state = app.get_state(config=config).values
                # Create a new state object with the updated prompt
                updated_state = {
                    **current_state,  # Keep all existing state values
                    "prompt": user_prompt,  # Update only the prompt
                }
                # Invoke with the complete updated state
                app.invoke(updated_state, config=config)

            # Every 5 interactions, check if we need to summarize
            if interaction_count % 5 == 0:
                print("\nManaging conversation history...")
                history_length = len(
                    app.get_state(config=config).values.get("conversation_history", [])
                )
                print(f"Conversation history contains {history_length} interactions")

        except KeyboardInterrupt:
            print("\nExiting Linux Assistant.")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
