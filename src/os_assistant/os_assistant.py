import traceback
import uuid

from os_assistant.utils import LOGGER

from .core.builder import build_assistant_graph
from .core.state import AssistantState
from .utils.settings import DOMAINS, GRAPH_VISUALIZE


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

                LOGGER.info(f"Graph visualization saved: {png_path}")
            except Exception as e:
                LOGGER.error(f"Graph visualization failed: {e}")

        LOGGER.info("Graph built successfully. Type 'exit' to quit.")
        LOGGER.debug(f"Session ID: {self.session_thread_id}")

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
            LOGGER.debug("Initialized assistant.")
        else:
            current_state = self.app.get_state(config=self.config).values
            updated_state = {
                **current_state,
                "prompt": prompt,
            }
            LOGGER.debug(f"BEFORE INVOKE - Updating state with prompt: {prompt}")
            LOGGER.debug(
                f"Current state keys: {current_state.keys() if hasattr(current_state, 'keys') else 'No keys'}"
            )
            self.app.invoke(updated_state, config=self.config)

        if self.interaction_count % 5 == 0:
            LOGGER.info("\nManaging conversation history...")
            history = self.app.get_state(config=self.config).values.get(
                "conversation_history", []
            )
            LOGGER.debug(f"Conversation history contains {len(history)} interactions")

    def run(self):
        while True:
            try:
                user_prompt = input("\nEnter your query: ")
                if user_prompt.lower() == "exit":
                    LOGGER.info("Exiting OS Assistant.")
                    break
                if not user_prompt.strip():
                    continue
                self.process_prompt(user_prompt)
            except KeyboardInterrupt:
                LOGGER.info("\nExiting OS Assistant.")
                break
            except Exception as e:
                LOGGER.exception(f"\nAn unexpected error occurred: {e}")
                traceback.print_exc()
