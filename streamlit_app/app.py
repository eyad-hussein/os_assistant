import sys
from pathlib import Path

# ========= Path setup: allow imports from src/os_assistant =========
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Now we can import your project
from os_assistant.os_assistant import OSAssistant  # type: ignore

import streamlit as st


# ========= Helper: get a single assistant instance per session =========
def get_assistant() -> OSAssistant:
    if "assistant" not in st.session_state:
        st.session_state.assistant = OSAssistant()
    return st.session_state.assistant


def run_assistant(prompt: str):
    """
    Call OSAssistant, then extract a clean answer string
    and some metadata from the final_result.
    Returns (answer: str, metadata: dict).
    """
    assistant = get_assistant()

    # Send the prompt into your graph
    assistant.process_prompt(prompt)

    # Get latest state from the app
    state = assistant.app.get_state(config=assistant.config).values

    result = (
        state.get("final_result")
        or state.get("information_response")
        or state.get("command_response")
        or state.get("domain_analysis")
    )

    # If nothing useful, just return the whole state as debug
    if result is None:
        return (
            "No result found in state.",
            {"state_keys": list(state.keys())},
        )

    # ---- Try to extract a clean answer string ----
    # 1) If result has .response.answer (like your example)
    answer = None
    response_obj = getattr(result, "response", None)

    if response_obj is not None:
        # pydantic/dataclass: response.answer
        answer = getattr(response_obj, "answer", None)
        # or dict-like
        if answer is None and isinstance(response_obj, dict):
            answer = response_obj.get("answer")

    # 2) If result itself has .answer
    if answer is None:
        answer = getattr(result, "answer", None)

    # 3) If result is a dict with nested "response"
    if answer is None and isinstance(result, dict):
        inner = result.get("response")
        if isinstance(inner, dict):
            answer = inner.get("answer")

    # 4) Fallback to stringifying everything
    if answer is None:
        answer = str(result)

    # ---- Build metadata for the debug/expander ----
    metadata = {}

    # best-effort extraction of fields like query, domains, etc.
    for field_name in ["query", "domains", "response_type", "context_summary"]:
        value = getattr(result, field_name, None)
        if value is None and isinstance(result, dict):
            value = result.get(field_name)
        if value is not None:
            metadata[field_name] = value

    # include the raw result (converted to string) for debugging
    metadata["raw_result_repr"] = repr(result)

    return answer, metadata


# ========= Streamlit UI =========
st.set_page_config(page_title="OS Assistant UI", page_icon="🤖", layout="wide")

st.title("OS Assistant – Streamlit UI")

st.write(
    "Type your query below. The request will be passed to the OSAssistant graph "
    "using the same logic as the CLI interface."
)
import re


def prettify_text(s: str) -> str:
    """
    Convert escaped sequences like '\\n' into real newlines,
    normalize line endings, and keep markdown readable.
    """
    if s is None:
        return ""

    # If it's not a string, make it one
    if not isinstance(s, str):
        s = str(s)

    # Convert literal backslash-n to real newline, etc.
    # Only do this if it looks like the string contains escapes.
    if "\\n" in s or "\\t" in s or "\\r" in s:
        try:
            s = s.encode("utf-8").decode("unicode_escape")
        except Exception:
            # safe fallback
            s = (
                s.replace("\\r\\n", "\n")
                .replace("\\n", "\n")
                .replace("\\t", "\t")
                .replace("\\r", "\n")
            )

    # Normalize Windows newlines
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    # Remove trailing spaces on lines (optional, makes markdown nicer)
    s = "\n".join(line.rstrip() for line in s.split("\n"))

    return s


def looks_like_code(s: str) -> bool:
    if not s:
        return False
    return (
        any(token in s for token in ["def ", "class ", "import ", "{", "}", "=>", ";"])
        and "\n" in s
    )


# Simple chat-style text input
user_input = st.text_area("Your query:", height=140, placeholder="Ask something...")

col_run, col_clear = st.columns([1, 1])

with col_run:
    run_clicked = st.button("Run", type="primary")

with col_clear:
    clear_clicked = st.button("Reset Session")

if clear_clicked:
    # Reset the assistant completely
    if "assistant" in st.session_state:
        del st.session_state["assistant"]
    st.experimental_rerun()

if run_clicked:
    if not user_input.strip():
        st.warning("Please enter a query first.")
    else:
        with st.spinner("Processing with OSAssistant..."):
            answer, metadata = run_assistant(user_input)

        st.subheader("Answer")
        pretty_answer = prettify_text(answer)
        if pretty_answer.strip().startswith("```"):
            st.markdown(pretty_answer)
        elif looks_like_code(pretty_answer):
            st.code(pretty_answer)
        else:
            st.markdown(pretty_answer)

        with st.expander("Details (debug info)"):
            st.json(metadata)
