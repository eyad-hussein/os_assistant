from langchain.schema import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from dagent.core.nodes.helpers import (
    build_combined_context,
    build_tool_context_info,
    is_code_execution_enabled,
    should_force_direct_response,
    tools,
)
from dagent.core.state import AssistantState
from dagent.parsers.setup import (
    code_execute_parser,
    fixed_info_response_parser,
    info_response_parser,
    parse_with_fix_and_extract,
)
from dagent.prompts.prompt_loader import load_prompt
from dagent.pydantic_models.schemas import (
    InformationResponse,
    ToolExecutionDetails,
)
from dagent.utils import LOGGER
from dagent.utils.model_factory import model
from dagent.utils.settings import (
    MODEL_BASE_URL,
    MODEL_NAME,
)


def information_generator_node(state: AssistantState) -> AssistantState:
    """Generate an information response"""
    LOGGER.info("\nNODE: information_generator_node")
    state["tool_originating_node"] = None

    # Check if code tool is enabled
    code_tool_enabled = is_code_execution_enabled()
    LOGGER.info(
        f"Code tool {'enabled' if code_tool_enabled else 'disabled'} in current mode"
    )

    # Get current tool usage count
    tool_usage_count = state.get("tool_usage_count", 0)
    LOGGER.info(f"Current tool usage count: {tool_usage_count}")

    # Determine if we should force direct info generation
    force_info = should_force_direct_response(state)
    if force_info:
        reason = (
            "Tool disabled"
            if not code_tool_enabled
            else f"Tool used {tool_usage_count} times"
        )
        LOGGER.warning(f"Forcing information generation without tool. Reason: {reason}")

    # Build combined context
    combined_context = build_combined_context(state)

    # Build tool context info
    tool_context_info = build_tool_context_info(state, force_info)

    # Load prompt from YAML
    info_generator_yaml = load_prompt("information_generator_node")

    # Create system message with the system message from YAML
    system_message = info_generator_yaml["system_message"].format(
        format_instructions=info_response_parser.get_format_instructions(),
        tool_format_instructions=code_execute_parser.get_format_instructions(),
    )

    # Format the prompt with required variables
    prompt = info_generator_yaml["prompt"].format(
        prompt=state["prompt"],
        combined_context=combined_context,
        tool_context_info=tool_context_info,
    )

    # Set up messages with system instruction
    messages = [SystemMessage(content=system_message), HumanMessage(content=prompt)]

    # Use appropriate model based on code tool availability
    if code_tool_enabled and not force_info:
        # Create a tool-enabled model
        information_model = ChatOllama(
            model=MODEL_NAME, temperature=0, base_url=MODEL_BASE_URL
        ).bind_tools(tools=tools)

        # Use the tool-enabled model
        content = information_model.invoke(messages)
    else:
        # Use regular model without tools
        content = model.invoke(messages)

    # First check if this is a tool call by looking for specific patterns
    tool_calls = str(content.tool_calls if hasattr(content, "tool_calls") else content)
    LOGGER.debug(f"tool_calls: {tool_calls}")
    # Fallback to original pattern matching logic - only if not forcing info
    is_tool_call = False
    if not force_info and (
        '"name": "code_execute_tool"' in tool_calls
        or "'name': 'code_execute_tool'" in tool_calls
    ):
        is_tool_call = True
        LOGGER.info("Detected tool call pattern in response")

        # Try to extract the question from the response
        import json
        import re

        # Try to extract JSON from the response
        json_match = re.search(r"({.*})", tool_calls, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(1)
                # Simple replacement: assume single-quoted keys/strings can be swapped (use with caution for complex cases)
                json_str = json_str.replace("'", '"')
                tool_data = json.loads(json_str)
                if isinstance(tool_data, dict) and "question" in tool_data:
                    state["tool_question"] = tool_data["question"]
                    LOGGER.debug(f"Extracted tool question: {tool_data['question']}")
                    state["tool_originating_node"] = "information_generation_node"

                    # Update the tool usage count in state
                    tool_usage_count += 1
                    state["tool_usage_count"] = tool_usage_count
                    LOGGER.info(f"Tool usage count increased to: {tool_usage_count}")

                    return state
            except json.JSONDecodeError:
                LOGGER.warning("Found JSON-like content but couldn't parse it")

    # Check for tool_calls attribute if pattern matching didn't work
    if not force_info and hasattr(content, "tool_calls") and content.tool_calls:
        is_tool_call = True
        LOGGER.info("Detected tool_calls attribute")

        # Extract tool call information
        for tool_call in content.tool_calls:
            if tool_call.get("name") == "code_execute_tool":
                question = tool_call.get("args", {}).get("question", "")
                state["tool_question"] = question
                LOGGER.debug(f"Extracted tool question from tool_calls: {question}")

                # Update the tool usage count in state
                tool_usage_count += 1
                state["tool_usage_count"] = tool_usage_count
                LOGGER.info(f"Tool usage count increased to: {tool_usage_count}")

                break
        state["tool_originating_node"] = "information_generation_node"
        return state

    # Only try to parse as InformationResponse if we're sure it's not a tool call
    if not is_tool_call:
        try:
            # Parse the response
            info_response = parse_with_fix_and_extract(
                content, info_response_parser, fixed_info_response_parser
            )

            # Ensure the result is a Pydantic model instance
            if not isinstance(info_response, InformationResponse):
                info_response = InformationResponse.model_validate(info_response)

            # Add tool information if available - use structured format
            if state.get("tool_context"):
                # Create structured tool execution details
                info_response.tool_execution = ToolExecutionDetails(
                    question=state.get("tool_question"),
                    code=state.get("tool_code"),
                    raw_output=state.get("raw_tool_results"),
                    analysis=state.get("tool_analysis"),
                    success=True,
                )
                # Also set legacy fields for backward compatibility
                info_response.tool_breakdown = (
                    "I used system tools to gather this information:"
                )
                info_response.tool_results = state.get("raw_tool_results", "")
                info_response.tool_interpretation = state.get(
                    "tool_analysis",
                    "The above results helped me provide you with an accurate answer.",
                )

            # Ensure the answer is personalized if not already
            if not any(
                phrase in info_response.answer.lower()
                for phrase in ["your", "you", "on your", "in your"]
            ):
                # Fix string index out of range error with proper length checking
                if len(info_response.answer) >= 2:
                    info_response.answer = f"On your system, {info_response.answer[0].lower()}{info_response.answer[1:]}"
                elif len(info_response.answer) == 1:
                    info_response.answer = (
                        f"On your system, {info_response.answer.lower()}"
                    )
                else:
                    info_response.answer = "On your system, I couldn't find specific information related to your query."

            state["information_response"] = info_response
            LOGGER.info("Successfully generated information response")

        except Exception as e:
            LOGGER.error(f"Error in information generation: {str(e)}")
            fallback_answer = f"I'm having trouble finding specific information about '{state['prompt']}' on your system. Could you provide more details or try a different query?"
            fallback_info = InformationResponse(
                answer=fallback_answer,
                sources=["System analysis"],
                tool_execution=None,
                tool_breakdown=None,
                tool_results=None,
                tool_interpretation=None,
            )
            state["information_response"] = fallback_info

    # At the end of the function, verify the info was generated if forced
    if force_info and not state.get("information_response"):
        LOGGER.warning(
            "Forced information generation but no information was created. Using fallback."
        )
        # Create structured tool execution details for fallback
        tool_exec = (
            ToolExecutionDetails(
                question=state.get("tool_question"),
                code=state.get("tool_code"),
                raw_output=state.get("raw_tool_results"),
                analysis=state.get("tool_analysis"),
                success=False,
                error_message="The tool execution did not provide sufficient information to answer your question.",
            )
            if state.get("tool_context")
            else None
        )

        fallback_info = InformationResponse(
            answer=f"After {tool_usage_count} attempts to gather information, I couldn't generate a specific answer about '{state['prompt']}'. Could you please rephrase your question?",
            sources=["System analysis after multiple tool executions"],
            tool_execution=tool_exec,
            tool_breakdown=f"Used tools {tool_usage_count} times but could not generate an appropriate answer.",
            tool_results=state.get("raw_tool_results", "No tool results available."),
            tool_interpretation="The tool execution did not provide sufficient information to answer your question.",
        )
        state["information_response"] = fallback_info

    return state
