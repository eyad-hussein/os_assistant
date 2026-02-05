import json
import re

from langchain.schema import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from os_assistant.core.nodes.helpers import (
    build_combined_context,
    build_tool_context_info,
    is_code_execution_enabled,
    should_force_direct_response,
    tools,
)
from os_assistant.core.state import AssistantState
from os_assistant.parsers.setup import (
    code_execute_parser,
    command_response_parser,
    fixed_command_response_parser,
    parse_with_fix_and_extract,
)
from os_assistant.prompts.prompt_loader import load_prompt
from os_assistant.pydantic_models.schemas import CommandResponse, ToolExecutionDetails
from os_assistant.utils import LOGGER
from os_assistant.utils.model_factory import model
from os_assistant.utils.settings import (
    MODEL_BASE_URL,
    MODEL_NAME,
    MODEL_TYPE,
)

# ============================================================================
# Helper Functions
# ============================================================================


def _extract_tool_question_from_json(tool_calls_str: str) -> str | None:
    """Extract question from tool call JSON string."""
    json_match = re.search(r"({.*})", tool_calls_str, re.DOTALL)
    if not json_match:
        return None

    try:
        json_str = json_match.group(1)
        # Replace single quotes with double quotes for valid JSON
        json_str = json_str.replace("'", '"')
        tool_data = json.loads(json_str)

        if isinstance(tool_data, dict) and "question" in tool_data:
            return tool_data["question"]
    except json.JSONDecodeError:
        LOGGER.exception("Found JSON-like content but couldn't parse it")

    return None


def _extract_tool_question_from_attribute(content) -> str | None:
    """Extract question from content's tool_calls attribute."""
    if not hasattr(content, "tool_calls") or not content.tool_calls:
        return None

    for tool_call in content.tool_calls:
        if tool_call.get("name") == "code_execute_tool":
            return tool_call.get("args", {}).get("question", "")

    return None


def _is_tool_call_detected(content, force_command: bool) -> bool:
    """Check if the model response contains a tool call."""
    if force_command:
        return False

    tool_calls_str = str(
        content.tool_calls if hasattr(content, "tool_calls") else content
    )

    # Check for tool call patterns in string
    return (
        '"name": "code_execute_tool"' in tool_calls_str
        or "'name': 'code_execute_tool'" in tool_calls_str
    )


def _handle_tool_call(state: AssistantState, content, force_command: bool) -> bool:
    """Process tool call if detected in model response."""
    if force_command:
        return False

    # Check if tool call is detected
    if not _is_tool_call_detected(content, force_command):
        return False

    tool_calls_str = str(
        content.tool_calls if hasattr(content, "tool_calls") else content
    )
    LOGGER.info(f"tool_calls: {tool_calls_str}")
    LOGGER.info("Detected tool call in response")

    # Try to extract question - first from attribute, then from JSON string
    question = _extract_tool_question_from_attribute(content)
    if not question:
        question = _extract_tool_question_from_json(tool_calls_str)

    if question:
        state["tool_question"] = question
        LOGGER.info(f"Extracted tool question: {question}")
        _update_tool_usage_count(state)
        state["tool_originating_node"] = "command_generation_node"
        return True

    LOGGER.warning("Tool call detected but could not extract question")
    return False


def _update_tool_usage_count(state: AssistantState) -> None:
    """Increment and log tool usage count."""
    tool_usage_count = state.get("tool_usage_count", 0) + 1
    state["tool_usage_count"] = tool_usage_count
    LOGGER.info(f"Tool usage count increased to: {tool_usage_count}")


def _personalize_explanation(explanation: str) -> str:
    """Add personalization to command explanation if not already present."""
    if any(
        phrase in explanation.lower()
        for phrase in ["your", "you", "on your", "in your"]
    ):
        return explanation

    if len(explanation) >= 2:
        return f"On your specific system, {explanation[0].lower()}{explanation[1:]}"
    elif len(explanation) == 1:
        return f"On your specific system, {explanation.lower()}"
    else:
        return "On your specific system, this command performs the requested operation."


def _create_command_response(state: AssistantState, content) -> CommandResponse:
    """Parse model response and create CommandResponse object."""
    # Parse the response
    command_response = parse_with_fix_and_extract(
        content, command_response_parser, fixed_command_response_parser
    )

    # Ensure the result is a Pydantic model instance
    if not isinstance(command_response, CommandResponse):
        command_response = CommandResponse.model_validate(command_response)

    # Handle backwards compatibility with "explanation" field
    if hasattr(command_response, "explanation") and not hasattr(
        command_response, "what_command_does"
    ):
        command_response.what_command_does = command_response.explanation

    # Add tool information if available - use structured format
    if state.get("tool_context"):
        # Create structured tool execution details
        command_response.tool_execution = ToolExecutionDetails(
            question=state.get("tool_question"),
            code=state.get("tool_code"),
            raw_output=state.get("raw_tool_results"),
            analysis=state.get("tool_analysis"),
            success=True,
        )
        # Also set legacy fields for backward compatibility
        command_response.tool_breakdown = (
            "I used system tools to gather information for this command:"
        )
        command_response.tool_results = state.get("raw_tool_results", "")
        command_response.tool_interpretation = state.get(
            "tool_analysis", "Based on these results, I generated the command above."
        )

    # Personalize the explanation
    command_response.what_command_does = _personalize_explanation(
        command_response.what_command_does
    )

    return command_response


def _create_fallback_command(
    state: AssistantState, error_msg: str = None
) -> CommandResponse:
    """Create a fallback command when generation fails."""
    tool_usage_count = state.get("tool_usage_count", 0)

    if tool_usage_count > 0:
        # Create structured tool execution details for fallback
        tool_exec = (
            ToolExecutionDetails(
                question=state.get("tool_question"),
                code=state.get("tool_code"),
                raw_output=state.get("raw_tool_results"),
                analysis=state.get("tool_analysis"),
                success=False,
                error_message="The tool execution did not provide sufficient information to generate a command.",
            )
            if state.get("tool_context")
            else None
        )

        return CommandResponse(
            command="echo 'Could not generate a specific command despite multiple tool executions'",
            what_command_does=f"After {tool_usage_count} attempts to gather information, I was unable to generate a precise command for '{state['prompt']}'.",
            security_notes="This is a fallback command due to generation difficulties.",
            tool_execution=tool_exec,
            tool_breakdown=f"Used tools {tool_usage_count} times but could not generate appropriate command.",
            tool_results=state.get("raw_tool_results", "No tool results available."),
            tool_interpretation="The tool execution did not provide sufficient information to generate a command.",
        )
    else:
        return CommandResponse(
            command="echo 'Could not generate a specific command for your request'",
            what_command_does=f"I was unable to generate a precise command for '{state['prompt']}' based on your system context.",
            security_notes="Please review any command carefully before execution.",
            tool_execution=None,
            tool_breakdown=None,
            tool_results=None,
            tool_interpretation=None,
        )


def _prepare_messages(state: AssistantState, force_command: bool) -> list:
    """Prepare system and user messages for the model."""
    # Build combined context
    combined_context = build_combined_context(state)

    # Build tool context info
    tool_context_info = build_tool_context_info(state, force_command)

    # Load prompt from YAML
    command_generator_yaml = load_prompt("command_generator_node")

    # Create system message
    system_message = command_generator_yaml["system_message"].format(
        format_instructions=command_response_parser.get_format_instructions(),
        tool_format_instructions=code_execute_parser.get_format_instructions(),
    )

    # Format the user prompt
    prompt = command_generator_yaml["prompt"].format(
        prompt=state["prompt"],
        domains=", ".join(
            state["domain_analysis"].domains
            if state.get("domain_analysis")
            else state.get("domains", [])
        ),
        combined_context=combined_context,
        tool_context_info=tool_context_info,
    )

    return [SystemMessage(content=system_message), HumanMessage(content=prompt)]


def _invoke_model(messages: list, code_tool_enabled: bool, force_command: bool):
    """Invoke the appropriate model based on tool availability."""
    if code_tool_enabled and not force_command:
        # Only use ChatOllama when the configured model provider is Ollama
        if MODEL_TYPE and MODEL_TYPE.upper() == "OLLAMA" and MODEL_BASE_URL:
            # Create a tool-enabled Ollama model
            command_model = ChatOllama(
                model=MODEL_NAME, temperature=0, base_url=MODEL_BASE_URL
            ).bind_tools(tools=tools)
            return command_model.invoke(messages)
        else:
            # For non-Ollama setups (e.g., OpenAI), fall back to the configured model
            # Note: tool-binding may not be available for all providers
            return model.invoke(messages)
    else:
        # Use regular model without tools
        return model.invoke(messages)


# ============================================================================
# Main Node Function
# ============================================================================


def command_generator_node(state: AssistantState) -> AssistantState:
    """Generate a command response by checking tools, preparing messages, handling tool calls, parsing responses, and providing fallbacks."""
    LOGGER.info("\nNODE: command_generator_node")
    state["tool_originating_node"] = None

    # ========================================================================
    # Step 1: Initialize and check tool availability
    # ========================================================================
    code_tool_enabled = is_code_execution_enabled()
    tool_usage_count = state.get("tool_usage_count", 0)
    force_command = should_force_direct_response(state)

    LOGGER.debug(
        f"Code tool {'enabled' if code_tool_enabled else 'disabled'} in current mode"
    )
    LOGGER.debug(f"Current tool usage count: {tool_usage_count}")

    if force_command:
        reason = (
            "Tool disabled"
            if not code_tool_enabled
            else f"Tool used {tool_usage_count} times"
        )
        LOGGER.warning(f"Forcing command generation without tool. Reason: {reason}")

    # ========================================================================
    # Step 2: Prepare messages and invoke model
    # ========================================================================
    messages = _prepare_messages(state, force_command)
    content = _invoke_model(messages, code_tool_enabled, force_command)

    # ========================================================================
    # Step 3: Detect tool request and exit so the graph can run tool nodes
    # ========================================================================
    if _handle_tool_call(state, content, force_command):
        return state

    # ========================================================================
    # Step 4: Parse command response from model
    # ========================================================================
    try:
        command_response = _create_command_response(state, content)
        state["command_response"] = command_response
        LOGGER.info(f"Generated command: {command_response.command}")

    except Exception as e:
        LOGGER.exception(f"Error generating command: {str(e)}")
        state["command_response"] = _create_fallback_command(state)

    # ========================================================================
    # Step 5: Verify command was generated if forced
    # ========================================================================
    if force_command and not state.get("command_response"):
        LOGGER.warning(
            "Forced command generation but no command was created. Using fallback."
        )
        state["command_response"] = _create_fallback_command(state)

    return state
