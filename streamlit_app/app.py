import re

import streamlit as st

from dagent.os_assistant import OSAssistant  # noqa: E402


# ========= Helper: get a single assistant instance per session =========
def get_assistant() -> OSAssistant:
    if "assistant" not in st.session_state:
        st.session_state.assistant = OSAssistant()
    return st.session_state.assistant


def run_assistant(prompt: str):
    """
    Call OSAssistant, then extract structured result data.
    Returns (result_type: str, result_data: dict, raw_state: dict).
    """
    assistant = get_assistant()

    # Send the prompt into your graph
    assistant.process_prompt(prompt)

    # Get latest state from the app
    state = assistant.app.get_state(config=assistant.config).values

    # Try to get results in order of preference
    final_result = state.get("final_result")
    information_response = state.get("information_response")
    command_response = state.get("command_response")
    domain_analysis = state.get("domain_analysis")

    # Convert Pydantic models to dicts for easier handling
    def to_dict(obj):
        if obj is None:
            return None
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "dict"):
            return obj.dict()
        if isinstance(obj, dict):
            return obj
        return None

    result_type = None
    result_data = None

    if final_result is not None:
        result_type = "final_result"
        result_data = to_dict(final_result)
    elif information_response is not None:
        result_type = "information_response"
        result_data = to_dict(information_response)
    elif command_response is not None:
        result_type = "command_response"
        result_data = to_dict(command_response)
    elif domain_analysis is not None:
        result_type = "domain_analysis"
        result_data = to_dict(domain_analysis)

    raw_state = {k: repr(v) for k, v in state.items()}

    return result_type, result_data, raw_state


# ========= Streamlit UI =========
st.set_page_config(page_title="DAgent UI", page_icon="🤖", layout="wide")

st.title("DAgent – UI")

st.write(
    "Type your query below. The request will be passed to the DAgent graph "
    "using the same logic as the CLI interface."
)


def prettify_text(s: str) -> str:
    """
    Convert escaped sequences like '\\n' into real newlines,
    normalize line endings, and keep markdown readable.
    Preserves Windows paths like E:\test.
    """
    if s is None:
        return ""

    # If it's not a string, make it one
    if not isinstance(s, str):
        s = str(s)

    # Only convert escaped newlines/carriage returns, NOT tabs
    # This preserves Windows paths like E:\test
    # Replace literal escaped newlines (\\n -> \n, \\r -> \r)
    # But be careful not to affect actual backslashes in paths

    # Handle double-escaped sequences first (from JSON serialization)
    s = s.replace("\\\\n", "\n")
    s = s.replace("\\\\r", "\r")

    # Handle single-escaped newlines (but not \t which could be in paths)
    # Only replace \n and \r that are clearly escape sequences
    # Use a more careful approach - only replace when followed by expected characters

    # Replace \r\n and \n that are escape sequences (not in paths)
    # Paths typically have single backslash followed by a letter, so \\n is safe to replace
    s = re.sub(r"(?<!\\)\\n", "\n", s)
    s = re.sub(r"(?<!\\)\\r", "\r", s)

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


# ========= Response Renderers =========
def render_tool_execution(tool_exec: dict):
    """Render structured tool execution details."""
    if not tool_exec:
        return

    question = tool_exec.get("question")
    code = tool_exec.get("code")
    raw_output = tool_exec.get("raw_output")
    analysis = tool_exec.get("analysis")
    success = tool_exec.get("success", True)
    error_message = tool_exec.get("error_message")

    st.markdown("### 🔧 Tool Execution Details")

    # Status indicator
    if success:
        st.success("✅ Tool execution completed successfully")
    else:
        st.error(f"❌ Tool execution failed: {error_message or 'Unknown error'}")

    # Question asked to the tool
    if question:
        with st.expander("❓ Question Asked", expanded=False):
            st.markdown(prettify_text(question))

    # Code that was executed
    if code:
        with st.expander("💻 Code Executed", expanded=True):
            # Try to detect language from code content
            if code.strip().startswith("import ") or "def " in code or "print(" in code:
                st.code(prettify_text(code), language="python")
            else:
                st.code(prettify_text(code), language="bash")

    # Raw output from execution
    if raw_output:
        with st.expander("📤 Execution Output", expanded=True):
            output_text = prettify_text(raw_output)
            # Display as code block for better formatting of command outputs
            st.code(output_text, language="text")

    # Analysis/interpretation
    if analysis:
        with st.expander("🔍 Analysis", expanded=False):
            st.markdown(prettify_text(analysis))


def render_legacy_tool_info(
    tool_breakdown: str, tool_results: str, tool_interpretation: str
):
    """Render legacy (unstructured) tool information."""
    st.markdown("### 🔧 Tool Details")

    if tool_breakdown:
        with st.expander("Tool Breakdown", expanded=False):
            st.markdown(prettify_text(tool_breakdown))

    if tool_results:
        with st.expander("Tool Results (Raw)", expanded=False):
            # Try to parse the old format and display nicely
            result_text = prettify_text(tool_results)

            # Check if it contains the old structured format
            if "Code used:" in result_text and "RAW EXECUTION RESULTS" in result_text:
                # Extract code section
                code_match = re.search(
                    r"Code used:\s*(.+?)(?=\n\s*=====|$)", result_text, re.DOTALL
                )
                if code_match:
                    st.markdown("**Code Executed:**")
                    st.code(code_match.group(1).strip(), language="python")

                # Extract raw results section
                raw_match = re.search(
                    r"RAW EXECUTION RESULTS.*?=====\s*(.+?)(?=\n\s*=====|Analysis:|$)",
                    result_text,
                    re.DOTALL,
                )
                if raw_match:
                    st.markdown("**Execution Output:**")
                    st.code(raw_match.group(1).strip(), language="text")

                # Extract analysis section
                analysis_match = re.search(
                    r"Analysis:\s*(.+?)(?=\n\s*IMPORTANT:|$)", result_text, re.DOTALL
                )
                if analysis_match:
                    st.markdown("**Analysis:**")
                    st.markdown(analysis_match.group(1).strip())
            else:
                # Just display as code block
                st.code(result_text)

    if tool_interpretation:
        with st.expander("Tool Interpretation", expanded=False):
            st.markdown(prettify_text(tool_interpretation))


def render_command_response(response: dict):
    """Render a CommandResponse in a structured, readable format."""
    command = response.get("command", "")
    is_python = response.get("is_python_script", False)
    what_it_does = response.get("what_command_does", "")
    security_notes = response.get("security_notes")
    tool_execution = response.get("tool_execution")
    # Legacy fields
    tool_breakdown = response.get("tool_breakdown")
    tool_results = response.get("tool_results")
    tool_interpretation = response.get("tool_interpretation")

    # Main command display
    st.markdown("### 💻 Generated Command")
    # lang = "python" if is_python else "powershell"
    # st.code(command, language=lang)
    st.markdown(prettify_text(command))
    if is_python:
        st.info("📜 This is a Python script that should be saved and executed.")

    # What the command does
    if what_it_does:
        st.markdown("### 📖 What This Command Does")
        st.markdown(prettify_text(what_it_does))

    # Security warnings
    if security_notes:
        st.markdown("### ⚠️ Security Notes")
        st.warning(prettify_text(security_notes))

    # Tool execution details - prefer structured format
    if tool_execution:
        render_tool_execution(tool_execution)
    elif tool_breakdown or tool_results or tool_interpretation:
        render_legacy_tool_info(tool_breakdown, tool_results, tool_interpretation)


def render_information_response(response: dict):
    """Render an InformationResponse in a structured, readable format."""
    answer = response.get("answer", "")
    sources = response.get("sources", [])
    tool_execution = response.get("tool_execution")
    # Legacy fields
    tool_breakdown = response.get("tool_breakdown")
    tool_results = response.get("tool_results")
    tool_interpretation = response.get("tool_interpretation")

    # Main answer
    st.markdown("### 📝 Answer")
    pretty_answer = prettify_text(answer)
    if pretty_answer.strip().startswith("```"):
        st.markdown(pretty_answer)
    else:
        st.markdown(pretty_answer)

    # Sources
    if sources:
        st.markdown("### 📚 Sources")
        for source in sources:
            st.markdown(f"- `{source}`")

    # Tool execution details - prefer structured format
    if tool_execution:
        render_tool_execution(tool_execution)
    elif tool_breakdown or tool_results or tool_interpretation:
        render_legacy_tool_info(tool_breakdown, tool_results, tool_interpretation)


def render_domain_analysis(analysis: dict):
    """Render a DomainAnalysis in a structured, readable format."""
    domains = analysis.get("domains", [])
    confidence = analysis.get("confidence", 0)
    reasoning = analysis.get("reasoning", "")
    requires_logs = analysis.get("requires_logs", False)

    st.markdown("### 🔍 Domain Analysis")

    # Domains as tags
    if domains:
        st.markdown("**Identified Domains:**")
        cols = st.columns(min(len(domains), 4))
        for i, domain in enumerate(domains):
            with cols[i % 4]:
                st.markdown(f"🏷️ `{domain}`")

    # Confidence meter
    st.markdown("**Confidence:**")
    st.progress(confidence)
    st.caption(f"{confidence * 100:.1f}%")

    # Reasoning
    if reasoning:
        st.markdown("**Reasoning:**")
        st.markdown(prettify_text(reasoning))

    # Requires logs indicator
    if requires_logs:
        st.info("📊 This query requires historical logs from the identified domains.")


def render_context_retrieval(context_retrieval: dict):
    """Render context retrieval details (SQL/RAG/Hybrid)."""
    if not context_retrieval:
        return

    query_intent = context_retrieval.get("query_intent")
    retrieval_sources = context_retrieval.get("retrieval_sources", [])
    sql_context = context_retrieval.get("sql_context")
    sql_query = context_retrieval.get("sql_query")
    sql_row_count = context_retrieval.get("sql_row_count", 0)
    rag_context = context_retrieval.get("rag_context")
    rag_doc_count = context_retrieval.get("rag_doc_count", 0)
    combined_context = context_retrieval.get("combined_context")
    domains_processed = context_retrieval.get("domains_processed", [])

    st.markdown("### 🔍 Context Retrieval Details")

    # Query Intent Badge
    if query_intent:
        intent_colors = {
            "structured": "🗃️",
            "semantic": "🧠",
            "hybrid": "🔀",
        }
        intent_icon = intent_colors.get(query_intent.lower(), "❓")
        st.markdown(f"**Query Intent:** {intent_icon} `{query_intent}`")

    # Retrieval Sources as tags
    if retrieval_sources:
        sources_str = " • ".join([f"`{src}`" for src in retrieval_sources])
        st.markdown(f"**Sources Used:** {sources_str}")

    # Statistics row
    col1, col2, col3 = st.columns(3)
    with col1:
        if sql_row_count > 0:
            st.metric("SQL Rows", sql_row_count)
        else:
            st.metric("SQL Rows", "-")
    with col2:
        if rag_doc_count > 0:
            st.metric("RAG Documents", rag_doc_count)
        else:
            st.metric("RAG Documents", "-")
    with col3:
        st.metric("Domains", len(domains_processed))

    # SQL Context section
    if sql_context:
        with st.expander("🗃️ SQL Database Context", expanded=False):
            if sql_query:
                st.markdown("**Query Executed:**")
                st.code(sql_query, language="sql")
            st.markdown("**Results:**")
            # Display SQL context as code for better formatting
            st.code(prettify_text(sql_context), language="text")

    # RAG Context section
    if rag_context and rag_context != sql_context:
        with st.expander("🧠 RAG Semantic Search Context", expanded=False):
            st.markdown(prettify_text(rag_context))

    # Combined Context section (only if different from individual contexts)
    if (
        combined_context
        and combined_context != sql_context
        and combined_context != rag_context
    ):
        with st.expander("🔀 Combined Context (Fused)", expanded=False):
            st.markdown(prettify_text(combined_context))


def render_final_result(result: dict):
    """Render a FinalResult with full context and nested response."""
    query = result.get("query", "")
    domains = result.get("domains", [])
    response_type = result.get("response_type", "")
    response = result.get("response", {})
    context_summary = result.get("context_summary", "")
    context_retrieval = result.get("context_retrieval")

    # Query summary header
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**Query:** {query}")
    with col2:
        badge_color = "🟢" if response_type == "information" else "🔵"
        st.markdown(f"{badge_color} **Type:** `{response_type}`")

    # Domains
    if domains:
        domain_str = " • ".join([f"`{d}`" for d in domains])
        st.markdown(f"**Domains:** {domain_str}")

    st.markdown("---")

    # Render context retrieval details (SQL/RAG/Hybrid)
    if context_retrieval:
        render_context_retrieval(context_retrieval)
        st.markdown("---")

    # Render the nested response based on type
    if response_type == "command":
        render_command_response(response)
    elif response_type == "information":
        render_information_response(response)
    else:
        # Fallback for unknown response types
        st.json(response)

    # Context summary at the bottom
    if context_summary:
        with st.expander("📋 Context Summary", expanded=False):
            st.markdown(prettify_text(context_summary))


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
    st.rerun()

if run_clicked:
    if not user_input.strip():
        st.warning("Please enter a query first.")
    else:
        with st.spinner("Processing with DAgent..."):
            result_type, result_data, raw_state = run_assistant(user_input)

        if result_type is None or result_data is None:
            st.error("No result found in state.")
            with st.expander("Debug: Raw State"):
                st.json(raw_state)
        else:
            # Render based on result type
            if result_type == "final_result":
                render_final_result(result_data)
            elif result_type == "information_response":
                render_information_response(result_data)
            elif result_type == "command_response":
                render_command_response(result_data)
            elif result_type == "domain_analysis":
                render_domain_analysis(result_data)
            else:
                # Unknown type fallback
                st.subheader("Result")
                st.json(result_data)

            # Debug expander with raw data
            with st.expander("🐛 Debug Info", expanded=False):
                st.markdown(f"**Result Type:** `{result_type}`")
                st.markdown("**Structured Data:**")
                st.json(result_data)
                st.markdown("**Raw State Keys:**")
                st.json(list(raw_state.keys()))
