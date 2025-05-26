import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from os_assistant.os_assistant import OSAssistant

if __name__ == "__main__":
    assistant = OSAssistant()
    assistant.run()
