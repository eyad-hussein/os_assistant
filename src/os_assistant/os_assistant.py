import traceback
import uuid

from .config.settings import DOMAINS
from .graph.builder import build_linux_assistant_graph
from .graph.state import LinuxAssistantState
from .utils.graph_visualizer import mermaid_to_png


class OSAssistant:
    def __init__(self):
        self.app = build_linux_assistant_graph()
        self.session_thread_id = str(uuid.uuid4())
        self.config = {"configurable": {"thread_id": self.session_thread_id}}
        self.interaction_count = 0
        self.initialized = False

        try:
            graph = self.app.get_graph()
            mermaid_txt = graph.draw_mermaid()
            png_path = mermaid_to_png(mermaid_txt)
            print("Generated", png_path)
            print("Graph visualization saved to linux_assistant_graph.png")
        except Exception as e:
            print(f"Note: Graph visualization could not be generated: {e}")
            print(
                "This is non-critical and the assistant will still function correctly."
            )

        print("Graph built successfully. Type 'exit' to quit.")
        print(f"Session ID: {self.session_thread_id}")

    def process_prompt(self, prompt: str):
        self.interaction_count += 1

        if not self.initialized:
            initial_state: LinuxAssistantState = {
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
            }
            self.app.invoke(initial_state, config=self.config)
            self.initialized = True
        else:
            current_state = self.app.get_state(config=self.config).values
            updated_state = {
                **current_state,
                "prompt": prompt,
            }
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
