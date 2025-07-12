import traceback
import uuid

from .utils.settings import DOMAINS, GRAPH_VISUALIZE
from .core.builder import build_assistant_graph
from .core.state import AssistantState


class OSAssistant:
    def __init__(self):
        self.app = build_assistant_graph()
        self.session_thread_id = str(uuid.uuid4())
        self.config = {"configurable": {"thread_id": self.session_thread_id}}
        self.interaction_count = 0
        self.initialized = False
        
        if GRAPH_VISUALIZE:
            try:
                from .utils.graph_visualizer import mermaid_to_png

                graph = self.app.get_graph()
                mermaid_txt = graph.draw_mermaid()
                png_path = mermaid_to_png(mermaid_txt)

                print(f"Graph visualization saved: {png_path}")
            except Exception as e:
                print(f"Graph visualization failed: {e}")

        print("Graph built successfully. Type 'exit' to quit.")
        print(f"Session ID: {self.session_thread_id}")

    def process_prompt(self, prompt: str):
        self.interaction_count += 1

        if not self.initialized:
            initial_state: AssistantState = {
                "prompt": prompt,
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
                "tool_usage_count": 0,
            }
            self.app.invoke(initial_state, config=self.config)
            self.initialized = True
        else:
            current_state = self.app.get_state(config=self.config).values
            updated_state = {
                **current_state,
                "prompt": prompt,
            }
            print(f"BEFORE INVOKE - Updating state with prompt: {prompt}")
            print(
                f"Current state keys: {current_state.keys() if hasattr(current_state, 'keys') else 'No keys'}"
            )
            self.app.invoke(updated_state, config=self.config)

        if self.interaction_count % 5 == 0:
            print("\nManaging conversation history...")
            history = self.app.get_state(config=self.config).values.get(
                "conversation_history", []
            )
            print(f"Conversation history contains {len(history)} interactions")

    def run(self):
        while True:
            try:
                user_prompt = input("\nEnter your query: ")
                if user_prompt.lower() == "exit":
                    print("Exiting OS Assistant.")
                    break
                if not user_prompt.strip():
                    continue
                self.process_prompt(user_prompt)
            except KeyboardInterrupt:
                print("\nExiting OS Assistant.")
                break
            except Exception as e:
                print(f"\nAn unexpected error occurred: {e}")
                traceback.print_exc()
