from langchain.schema import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from os_assistant.utils.model_factory import model
from os_assistant.core.state import AssistantState
from os_assistant.prompts.prompt_loader import load_prompt
from os_assistant.pydantic_models.schemas import CommandResponse
from os_assistant.core.nodes.helpers import (
    is_code_execution_enabled,
    build_combined_context,
    should_force_direct_response,
    build_tool_context_info,
    tools
)
from os_assistant.parsers.setup import (
    code_execute_parser,
    command_response_parser,
    fixed_command_response_parser,
    parse_with_fix_and_extract,
)
from os_assistant.utils.settings import (
    MODEL_BASE_URL,
    MODEL_NAME,
)

def command_generator_node(state: AssistantState) -> AssistantState:
    """Generate a command response"""
    print("\nNODE: command_generator_node")
    state["tool_originating_node"] = None

    # Check if code tool is enabled
    code_tool_enabled = is_code_execution_enabled()
    print(f"Code tool {'enabled' if code_tool_enabled else 'disabled'} in current mode")

    # Get current tool usage count
    tool_usage_count = state.get("tool_usage_count", 0)
    print(f"Current tool usage count: {tool_usage_count}")

    # Determine if we should force direct command generation
    force_command = should_force_direct_response(state)
    if force_command:
        reason = (
            "Tool disabled"
            if not code_tool_enabled
            else f"Tool used {tool_usage_count} times"
        )
        print(f"Forcing command generation without tool. Reason: {reason}")

    # Build combined context
    combined_context = build_combined_context(state)

    # Build tool context info
    tool_context_info = build_tool_context_info(state, force_command)

    # Load prompt from YAML
    command_generator_yaml = load_prompt("command_generator_node")

    # Create system message with the system message from YAML
    system_message = command_generator_yaml["system_message"].format(
        format_instructions=command_response_parser.get_format_instructions(),
        tool_format_instructions=code_execute_parser.get_format_instructions(),
    )

    # Format the prompt with required variables
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

    # Set up messages with system instruction
    messages = [SystemMessage(content=system_message), HumanMessage(content=prompt)]

    # Use appropriate model based on code tool availability
    if code_tool_enabled and not force_command:
        # Create a tool-enabled model
        command_model = ChatOllama(
            model=MODEL_NAME, temperature=0, base_url=MODEL_BASE_URL
        ).bind_tools(tools=tools)

        # Use the tool-enabled model
        content = command_model.invoke(messages)
    else:
        # Use regular model without tools
        content = model.invoke(messages)

    # First check if this is a tool call by looking for specific patterns
    tool_calls = str(content.tool_calls if hasattr(content, "tool_calls") else content)
    print(f"tool_calls: {tool_calls}")

    # original pattern matching logic
    is_tool_call = False
    if not force_command and (
        '"name": "code_execute_tool"' in tool_calls
        or "'name': 'code_execute_tool'" in tool_calls
    ):
        is_tool_call = True
        print("Detected tool call pattern in response")

        # Try to extract the question from the response
        import json
        import re

        # Try to extract JSON from the response
        json_match = re.search(r"({.*})", tool_calls, re.DOTALL)
        if json_match:
            try:
                tool_data = json.loads(json_match.group(1))
                if isinstance(tool_data, dict) and "question" in tool_data:
                    state["tool_question"] = tool_data["question"]
                    print(f"Extracted tool question: {tool_data['question']}")
                    state["tool_originating_node"] = "command_generation_node"

                    # Update the tool usage count in state
                    tool_usage_count += 1
                    state["tool_usage_count"] = tool_usage_count
                    print(f"Tool usage count increased to: {tool_usage_count}")

                    return state
            except json.JSONDecodeError:
                print("Found JSON-like content but couldn't parse it")

    # Check for tool_calls attribute if pattern matching didn't work
    if not force_command and hasattr(content, "tool_calls") and content.tool_calls:
        is_tool_call = True
        print("Detected tool_calls attribute")

        # Extract tool call information
        for tool_call in content.tool_calls:
            if tool_call.get("name") == "code_execute_tool":
                question = tool_call.get("args", {}).get("question", "")
                state["tool_question"] = question
                print(f"Extracted tool question from tool_calls: {question}")

                # Update the tool usage count in state
                tool_usage_count += 1
                state["tool_usage_count"] = tool_usage_count
                print(f"Tool usage count increased to: {tool_usage_count}")

                break
        state["tool_originating_node"] = "command_generation_node"
        return state

    # Only try to parse as CommandResponse if we're sure it's not a tool call
    if not is_tool_call:
        try:
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

            # Add tool information if available
            # TODO: Change it to our new idea (combine the three ideas)
            if state.get("tool_context"):
                command_response.tool_breakdown = (
                    "I used system tools to gather information for this command:"
                )
                command_response.tool_results = state.get("tool_context", "")
                command_response.tool_interpretation = (
                    "Based on these results, I generated the command above."
                )

            # Ensure the explanation is personalized if not already
            if not any(
                phrase in command_response.what_command_does.lower()
                for phrase in ["your", "you", "on your", "in your"]
            ):
                # Fix string index out of range error with proper length checking
                if len(command_response.what_command_does) >= 2:
                    command_response.what_command_does = f"On your specific system, {command_response.what_command_does[0].lower()}{command_response.what_command_does[1:]}"
                elif len(command_response.what_command_does) == 1:
                    command_response.what_command_does = f"On your specific system, {command_response.what_command_does.lower()}"
                else:
                    command_response.what_command_does = "On your specific system, this command performs the requested operation."

            state["command_response"] = command_response

            print(f"Generated command: {command_response.command}")

        except Exception as e:
            print(f"Error generating command: {str(e)}")
            # Fallback command
            fallback_command = CommandResponse(
                command="echo 'Could not generate a specific command for your request'",
                what_command_does=f"I was unable to generate a precise command for '{state['prompt']}' based on your system context.",
                security_notes="Please review any command carefully before execution.",
                tool_breakdown=None,
                tool_results=None,
                tool_interpretation=None,
            )
            state["command_response"] = fallback_command

    # At the end of the function, verify the command was generated if forced
    if force_command and not state.get("command_response"):
        print(
            "WARNING: Forced command generation but no command was created. Using fallback."
        )
        fallback_command = CommandResponse(
            command="echo 'Could not generate a specific command despite multiple tool executions'",
            what_command_does=f"After {tool_usage_count} attempts to gather information, I was unable to generate a precise command for '{state['prompt']}'.",
            security_notes="This is a fallback command due to generation difficulties.",
            tool_breakdown=f"Used tools {tool_usage_count} times but could not generate appropriate command.",
            tool_results=state.get("tool_context", "No tool results available."),
            tool_interpretation="The tool execution did not provide sufficient information to generate a command.",
        )
        state["command_response"] = fallback_command

    return state