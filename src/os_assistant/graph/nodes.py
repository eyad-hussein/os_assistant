from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import yaml
from langchain.schema import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from tracer.config import LogDomain

from os_assistant.config.settings import DOMAINS, MODEL_BASE_URL, MODEL_NAME, model
from os_assistant.parsers.setup import (
    code_execute_parser,
    command_response_parser,
    domain_analysis_parser,
    fixed_code_execute_parser,
    fixed_command_response_parser,
    fixed_domain_analysis_parser,
    fixed_info_response_parser,
    fixed_query_type_parser,
    info_response_parser,
    parse_with_fix_and_extract,
    query_type_parser,
)
from os_assistant.pydantic_models.schemas import (
    CodeExecuteRequest,
    CommandResponse,
    DomainAnalysis,
    FinalResult,
    InformationResponse,
    QueryTypeResult,
)
from os_assistant.tools.agentic_rag.application.search import search_logs
from os_assistant.tools.code_agent.wrapper import code_execute_tool

if TYPE_CHECKING:
    from os_assistant.graph.state import LinuxAssistantState


# Function to load prompts from YAML files
def load_prompt(prompt_name):
    """Load a prompt from a YAML file."""
    prompt_path = f"src/os_assistant/prompts/{prompt_name}.yaml"
    with open(prompt_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --- Node Functions ---
tools = [code_execute_tool]


def initialize_state(state: LinuxAssistantState, prompt: str) -> LinuxAssistantState:
    """Initialize the state with user prompt"""
    print("\nNODE: initialize_state")
    state["prompt"] = prompt
    state["domains"] = DOMAINS  # Use domains from config
    state["contexts"] = {}
    state["domains_to_process"] = []
    state["current_domain"] = None
    state["domain_analysis"] = None
    state["query_type"] = None
    state["command_response"] = None
    state["information_response"] = None
    state["final_result"] = None
    state["tool_usage_count"] = 0
    return state


def domain_analysis_node(state: LinuxAssistantState) -> LinuxAssistantState:
    """Analyze which domains are relevant to the query"""
    print("\nNODE: domain_analysis_node")

    print("\nAnalyzing query domains...")

    # Load prompt from YAML
    domain_analysis_yaml = load_prompt("domain_analysis_node")

    # Format the prompt with required variables
    prompt = domain_analysis_yaml["prompt"].format(
        prompt=state["prompt"],
        domains=", ".join(state["domains"]),
        format_instructions=domain_analysis_parser.get_format_instructions(),
    )

    messages = [HumanMessage(content=prompt)]
    content = model.invoke(messages)

    try:
        # Use the helper function for parsing attempts
        domain_analysis = parse_with_fix_and_extract(
            content, domain_analysis_parser, fixed_domain_analysis_parser
        )

        # Ensure the result is a Pydantic model instance before accessing attributes
        if not isinstance(domain_analysis, DomainAnalysis):
            # If parsing/fixing returned raw dict, try validating it
            domain_analysis = DomainAnalysis.model_validate(domain_analysis)

        state["domain_analysis"] = domain_analysis
        state["domains_to_process"] = (
            domain_analysis.domains.copy()
        )  # Use identified domains

        print(f"Domains identified: {domain_analysis.domains}")
        print(f"Confidence: {domain_analysis.confidence}")
        print(f"Reasoning: {domain_analysis.reasoning}")

    except Exception as e:
        print(f"Error analyzing domains: {str(e)}")
        # Fallback to using all domains
        fallback_analysis = DomainAnalysis(
            domains=state["domains"],  # Use all available domains
            confidence=0.5,
            reasoning=f"Fallback: using all available domains due to analysis error for query: '{state['prompt']}'",
        )
        state["domain_analysis"] = fallback_analysis
        state["domains_to_process"] = state[
            "domains"
        ].copy()  # Use all available domains

    return state


def context_retrieval_node(state: LinuxAssistantState) -> LinuxAssistantState:
    """Retrieve context for a domain using Agentic_RAG search_logs"""
    print("\nNODE: context_retrieval_node")

    if not state["domains_to_process"]:
        print("No more domains to process for context retrieval.")
        return state  # No more domains to process

    current_domain = state["domains_to_process"].pop(0)
    state["current_domain"] = current_domain

    print(f"\nRetrieving context for domain: {current_domain}")

    try:
        # Convert domain string to LogDomain enum

        try:
            domain_enum = LogDomain(current_domain.strip())
        except KeyError:
            print(
                f"Warning: Domain {current_domain} not found in LogDomain enum. Using FS as fallback."
            )
            domain_enum = LogDomain.FS

        # Call search_logs from Agentic_RAG
        logs, summaries = search_logs(
            query=state["prompt"],
            domains=[domain_enum],
            top_k=3,  # Get top 3 results
            summarize=True,  # Get summaries too
            auto_init=True,  # Auto-initialize if needed
        )

        # Format the results into context for the state
        context = ""
        if logs:
            for i, log in enumerate(logs):
                domain_info = f"Domain: {log.get('domain', domain_enum.name)}\n"
                context += f"{domain_info}Log #{log['log_number']} (Timestamp: {log['timestamp']})\n"

                # Include summary if available
                if summaries and i < len(summaries):
                    context += f"Summary: {summaries[i]}\n"

                # Add the log text
                context += f"Content: {log['log_text']}\n\n"
        else:
            context = f"No relevant logs found for query: '{state['prompt']}' in domain {current_domain}"

        # Store the context
        state["contexts"][current_domain] = context
        print(f"Retrieved context from {current_domain} using Agentic_RAG")

    except Exception as e:
        print(f"Error retrieving context for {current_domain}: {str(e)}")
        state["contexts"][
            current_domain
        ] = f"Error retrieving context for {current_domain}: {str(e)}"

    # Clear current_domain after processing
    state["current_domain"] = None
    return state


def query_classifier_node(state: LinuxAssistantState) -> LinuxAssistantState:
    """Classify the query type (command or information)"""
    print("\nNODE: query_classifier_node")

    print("\nClassifying query type...")

    combined_context = ""
    # Use only contexts from the domains identified in the analysis step
    relevant_domains = (
        state["domain_analysis"].domains
        if state["domain_analysis"]
        else state["domains"]
    )
    for domain in relevant_domains:
        context = state["contexts"].get(domain, "No context retrieved.")
        combined_context += f"--- {domain.upper()} DOMAIN ---\n{context}\n\n"

    if not combined_context:
        combined_context = "No specific context was retrieved for the relevant domains."

    # Load prompt from YAML
    query_classifier_yaml = load_prompt("query_classifier_node")

    # Format the prompt with required variables
    prompt = query_classifier_yaml["prompt"].format(
        prompt=state["prompt"],
        combined_context=combined_context,
        format_instructions=query_type_parser.get_format_instructions(),
    )

    messages = [HumanMessage(content=prompt)]
    content = model.invoke(messages)

    try:
        # Use the helper function for parsing attempts
        query_type = parse_with_fix_and_extract(
            content, query_type_parser, fixed_query_type_parser
        )

        # Ensure the result is a Pydantic model instance
        if not isinstance(query_type, QueryTypeResult):
            query_type = QueryTypeResult.model_validate(query_type)

        state["query_type"] = query_type

        print(f"Query classified as: {query_type.query_type}")
        print(f"Reasoning: {query_type.reasoning}")

    except Exception as e:
        print(f"Error classifying query: {str(e)}")
        # Fallback to information type
        fallback_query_type = QueryTypeResult(
            query_type="information",
            reasoning=f"Fallback: defaulting to information type due to classification error for query: '{state['prompt']}'",
            confidence=0.5,
        )
        state["query_type"] = fallback_query_type

    return state


def command_generator_node(state: LinuxAssistantState) -> LinuxAssistantState:
    """Generate a command response"""
    print("\nNODE: command_generator_node")
    state["tool_originating_node"] = None

    # IMPORTANT: Always retrieve the current tool count from state
    tool_usage_count = state.get("tool_usage_count", 0)
    print(f"Current tool usage count: {tool_usage_count}")

    # Check if we've already used the tool 3 times - if so, force command generation
    force_command = tool_usage_count >= 3
    if force_command:
        print(
            f"Tool has been used {tool_usage_count} times. Forcing command generation."
        )

    combined_context = ""
    # Use only contexts from the domains identified in the analysis step
    relevant_domains = (
        state["domain_analysis"].domains
        if state["domain_analysis"]
        else state["domains"]
    )
    for domain in relevant_domains:
        context = state["contexts"].get(domain, "No context retrieved.")
        combined_context += f"--- {domain.upper()} DOMAIN ---\n{context}\n\n"

    if not combined_context:
        combined_context = "No specific context was retrieved for the relevant domains."

    # Add information about tool_context to the prompt
    tool_context_info = ""
    if state.get("tool_context"):
        tool_context_info = f"""
        IMPORTANT: I've already executed the tool for you! The results are below:
        
        {state["tool_context"]}
        
        Use this information to create an appropriate command.
        You can request additional information with the tool if needed.
        """

    # Modify the tool_context_info to be more explicit
    if force_command:
        tool_context_info += f"""
        CRITICAL INSTRUCTION: You have already used the tool {tool_usage_count} times.
        YOU MUST NOW GENERATE A COMMAND RESPONSE WITHOUT USING THE TOOL AGAIN.
        DO NOT REQUEST MORE INFORMATION - USE WHAT YOU HAVE TO GENERATE A COMMAND.
        """

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
        domains=", ".join(relevant_domains),
        combined_context=combined_context,
        tool_context_info=tool_context_info,
    )

    # Set up messages with system instruction
    messages = [SystemMessage(content=system_message), HumanMessage(content=prompt)]

    # Create a tool-enabled model
    command_model = ChatOllama(model=MODEL_NAME, base_url=MODEL_BASE_URL).bind_tools(
        tools=tools
    )

    # Use the tool-enabled model
    content = command_model.invoke(messages)

    print(f"Response type: {type(content)}")
    print("INFO:", content)

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

            # Ensure the explanation is personalized if not already
            if not any(
                phrase in command_response.explanation.lower()
                for phrase in ["your", "you", "on your", "in your"]
            ):
                command_response.explanation = f"On your specific system, {command_response.explanation[0].lower()}{command_response.explanation[1:]}"

            state["command_response"] = command_response

            print(f"Generated command: {command_response.command}")

        except Exception as e:
            print(f"Error generating command: {str(e)}")
            # Fallback command
            fallback_command = CommandResponse(
                command="echo 'Could not generate a specific command for your request'",
                explanation=f"I was unable to generate a precise command for '{state['prompt']}' based on your system context.",
                security_notes="Please review any command carefully before execution.",
            )
            state["command_response"] = fallback_command

    # At the end of the function, verify the command was generated if forced
    if force_command and not state.get("command_response"):
        print(
            "WARNING: Forced command generation but no command was created. Using fallback."
        )
        fallback_command = CommandResponse(
            command="echo 'Could not generate a specific command despite multiple tool executions'",
            explanation=f"After {tool_usage_count} attempts to gather information, I was unable to generate a precise command for '{state['prompt']}'.",
            security_notes="This is a fallback command due to generation difficulties.",
        )
        state["command_response"] = fallback_command

    return state


def tool_execution_node(state: LinuxAssistantState) -> LinuxAssistantState:
    """Execute a tool and store the results in the state"""
    print("\nNODE: tool_execution_node")
    print(f"State keys before execution: {state.keys()}")
    print(f"Tool usage count: {state.get('tool_usage_count', 0)}")

    # Extract the question from the state
    question = str(state.get("tool_question"))
    if not question:
        print("Error: No tool question found in state.")
        return state

    print(f"Tool question: {question}")

    try:
        # Execute the question
        tool_state = code_execute_tool(question)
        # Check if execution was aborted due to too many errors
        if "Too many consecutive errors" in (tool_state.get("error_code") or ""):
            print("Tool execution aborted: Too many consecutive errors")

            # Create an error message to include in the state
            error_message = f"""
            I attempted to execute code to answer your question, but encountered multiple errors.
            
            Question: {question}
            
            After 3 failed attempts, I had to abort execution for safety reasons.
            Please try simplifying your request or provide more specific instructions.
            """

            state["tool_context"] = error_message
            return state

        print("Tool execution completed successfully.")
        print(f"Code executed: {tool_state['code']}")
        print(
            f"Execution result: {tool_state['execution_result'][:100]}..."
            if len(tool_state["execution_result"]) > 100
            else f"Execution result: {tool_state['execution_result']}"
        )

        # Prepare a message to add to the state that will be used when returning to the originating node
        tool_context = f"""
        I used the code_execute_tool to answer your question.
        
        Question: {question}
        
        Code used: {tool_state["code"]}
        
        Execution result: {tool_state["execution_result"]}
        
        Analysis: {tool_state["agent_output"]}
        """

        state["tool_context"] = tool_context

    except Exception as e:
        print(f"Error executing tool: {str(e)}")
        state["tool_context"] = f"An error occurred while executing the tool: {str(e)}"

    print("EXITING tool_execution_node")
    print(f"Modified state keys: {state.keys()}")
    print(f"Prompt value: {state.get('prompt')}")
    return state


def information_generator_node(state: LinuxAssistantState) -> LinuxAssistantState:
    """Generate an information response"""
    print("\nNODE: information_generator_node")
    state["tool_originating_node"] = None

    # IMPORTANT: Always retrieve the current tool count from state
    tool_usage_count = state.get("tool_usage_count", 0)
    print(f"Current tool usage count: {tool_usage_count}")

    # Check if we've already used the tool 3 times - if so, force info generation
    force_info = tool_usage_count >= 3
    if force_info:
        print(
            f"Tool has been used {tool_usage_count} times. Forcing information generation."
        )

    combined_context = ""
    # Use only contexts from the domains identified in the analysis step
    relevant_domains = (
        state["domain_analysis"].domains
        if state["domain_analysis"]
        else state["domains"]
    )
    for domain in relevant_domains:
        context = state["contexts"].get(domain, "No context retrieved.")
        combined_context += f"--- {domain.upper()} DOMAIN ---\n{context}\n\n"

    if not combined_context:
        combined_context = "No specific context was retrieved for the relevant domains."

    # Add information about tool_context to the prompt
    tool_context_info = ""
    if state.get("tool_context"):
        tool_context_info = f"""
        IMPORTANT: I've already executed the tool for you! The results are below:
        
        {state["tool_context"]}
        
        Use this information to provide a comprehensive answer.
        You can request additional information with the tool if needed.
        """

    # Modify the tool_context_info to be more explicit
    if force_info:
        tool_context_info += f"""
        CRITICAL INSTRUCTION: You have already used the tool {tool_usage_count} times.
        YOU MUST NOW GENERATE AN INFORMATION RESPONSE WITHOUT USING THE TOOL AGAIN.
        DO NOT REQUEST MORE INFORMATION - USE WHAT YOU HAVE TO GENERATE AN ANSWER.
        """

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

    # Create a tool-enabled model
    information_model = ChatOllama(
        model=MODEL_NAME, base_url=MODEL_BASE_URL
    ).bind_tools(tools=tools)

    # IMPORTANT: Use the tool-enabled model (not the regular model)
    content = information_model.invoke(messages)

    print(f"Response type: {type(content)}")
    print("INFO:", content)

    # First check if this is a tool call by looking for specific patterns
    tool_calls = str(content.tool_calls if hasattr(content, "tool_calls") else content)
    print(f"tool_calls: {tool_calls}")
    # Fallback to original pattern matching logic - only if not forcing info
    is_tool_call = False
    if not force_info and (
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
                    state["tool_originating_node"] = "information_generation_node"

                    # Update the tool usage count in state
                    tool_usage_count += 1
                    state["tool_usage_count"] = tool_usage_count
                    print(f"Tool usage count increased to: {tool_usage_count}")

                    return state
            except json.JSONDecodeError:
                print("Found JSON-like content but couldn't parse it")

    # Check for tool_calls attribute if pattern matching didn't work
    if not force_info and hasattr(content, "tool_calls") and content.tool_calls:
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

            # Ensure the answer is personalized if not already
            if not any(
                phrase in info_response.answer.lower()
                for phrase in ["your", "you", "on your", "in your"]
            ):
                info_response.answer = f"On your system, {info_response.answer[0].lower()}{info_response.answer[1:]}"

            state["information_response"] = info_response
            print("Successfully generated information response")

        except Exception as e:
            print(f"Error in information generation: {str(e)}")
            fallback_answer = f"I'm having trouble finding specific information about '{state['prompt']}' on your system. Could you provide more details or try a different query?"
            fallback_info = InformationResponse(
                answer=fallback_answer, sources=["System analysis"]
            )
            state["information_response"] = fallback_info

    # At the end of the function, verify the info was generated if forced
    if force_info and not state.get("information_response"):
        print(
            "WARNING: Forced information generation but no information was created. Using fallback."
        )
        fallback_info = InformationResponse(
            answer=f"After {tool_usage_count} attempts to gather information, I couldn't generate a specific answer about '{state['prompt']}'. Could you please rephrase your question?",
            sources=["System analysis after multiple tool executions"],
        )
        state["information_response"] = fallback_info

    return state


def prepare_final_result_node(state: LinuxAssistantState) -> LinuxAssistantState:
    """Prepare the final result"""
    print("\nNODE: prepare_final_result_node")

    # Ensure domain_analysis and query_type exist before accessing keys
    domains_tmp = state.get("domain_analysis")
    if domains_tmp is None:
        print("Warning: Domain analysis missing, using all domains for final result.")
        domains = state["domains"]  # Fallback to all domains
    else:
        domains = domains_tmp.domains

    query_type_tmp = state.get("query_type")
    if query_type_tmp is None:
        print(
            "Warning: Query type missing, defaulting to 'information' for final result."
        )
        response_type = "information"  # Fallback type
    else:
        response_type = query_type_tmp.query_type

    # Create context summary
    context_summary = "Analyzed information from: "
    context_summary += ", ".join(domains)

    # Determine response content
    match response_type:
        case "command":
            if state.get("command_response"):
                response = state["command_response"]
            else:
                print("Warning: Command response expected but missing.")
                # Create a fallback command response if needed, or switch type
                response_type = "information"  # Switch to info if command failed
                response = InformationResponse(
                    answer=f"Could not generate a command for '{state['prompt']}'. Please try rephrasing.",
                    sources=["System processing error"],
                )
        # Handle information response (either primary or fallback)
        case "information":
            if state.get("information_response"):
                response = state["information_response"]
            else:
                print("Warning: Information response expected but missing.")
                # Create a fallback information response
                response = InformationResponse(
                    answer=f"Unable to generate an answer for '{state['prompt']}' based on the available information.",
                    sources=["System processing error"],
                )

    # Ensure response is not None before creating FinalResult
    if response is None:
        print("Error: Could not determine a valid response for the final result.")
        # Handle this case, maybe set final_result to an error state or raise exception
        # For now, create a minimal error response
        response = InformationResponse(
            answer="An unexpected error occurred while generating the response.",
            sources=["System error"],
        )
        response_type = "information"  # Ensure type matches the fallback

    # Create final result
    final_result = FinalResult(
        query=state["prompt"],
        domains=domains,
        response_type=response_type,  # Use the potentially updated response_type
        response=response,  # Pass the dictionary directly
        context_summary=context_summary,
    )

    state["final_result"] = final_result

    return state


def conversation_context_node(state: LinuxAssistantState) -> LinuxAssistantState:
    """Provide conversation context by analyzing history and refining the prompt"""
    print("\nNODE: conversation_context_node")

    print("\nAnalyzing conversation context...")

    # Access conversation history
    conversation_history = state.get("conversation_history", [])

    # If this is the first interaction, nothing to enhance
    if not conversation_history:
        print("No conversation history found. Processing original query.")
        return state

    # Get the current prompt and previous interactions
    current_prompt = state["prompt"]

    # Format conversation history for the LLM with ranking by relevance
    formatted_history = ""

    # Include the most recent 3-5 interactions, prioritizing those that seem most relevant
    recent_history = conversation_history[-5:]
    for idx, entry in enumerate(recent_history):
        query = entry.get("query", "N/A")

        # Format the response based on the type
        response = entry.get("response", {})
        if isinstance(response, dict):
            if entry.get("response_type") == "command":
                cmd = response.get("command", "N/A")
                explanation = response.get("explanation", "N/A")
                formatted_history += f"Interaction {idx + 1}:\nUser: {query}\nAssistant: I suggested this command: '{cmd}'\n{explanation}\n\n"
            elif entry.get("response_type") == "information":
                answer = response.get("answer", "N/A")
                formatted_history += (
                    f"Interaction {idx + 1}:\nUser: {query}\nAssistant: {answer}\n\n"
                )
        else:
            formatted_history += (
                f"Interaction {idx + 1}:\nUser: {query}\nAssistant: {str(response)}\n\n"
            )

    # Load prompt from YAML
    conversation_context_yaml = load_prompt("conversation_context_node")

    # Format the prompt with required variables
    context_prompt = conversation_context_yaml["prompt"].format(
        formatted_history=formatted_history, current_prompt=current_prompt
    )

    # Ask the model to enhance the query
    messages = [HumanMessage(content=context_prompt)]
    model_response = model.invoke(messages)

    # Convert AIMessage to string properly, handling different response formats
    if hasattr(model_response, "content"):
        refined_prompt = str(model_response.content)
    else:
        refined_prompt = str(model_response)

    # Clean up any potential formatting issues
    refined_prompt = refined_prompt.strip()
    if refined_prompt.startswith('"') and refined_prompt.endswith('"'):
        refined_prompt = refined_prompt[1:-1]

    # If the model returns something that looks like an explanation rather than a query,
    # or if the refined prompt isn't substantially different, use the original
    print("INFO:", refined_prompt)
    if (
        "I don't need to enhance" in refined_prompt
        or "The query is self-contained" in refined_prompt
        or refined_prompt == current_prompt
    ):
        print("Query is self-contained or refinement unsuccessful. Using original.")
        return state

    print(f"Original query: {current_prompt}")
    print(f"Enhanced query: {refined_prompt}")

    # Store both the original and refined prompts
    state["original_prompt"] = current_prompt
    state["prompt"] = refined_prompt

    return state


def display_result_node(state: LinuxAssistantState) -> LinuxAssistantState:
    """Display the final result to the user and record in conversation history"""
    print("\nNODE: display_result_node")

    if not state.get("final_result"):
        print("\nError: No final result generated.")
        return state

    final_result = state["final_result"]
    assert final_result is not None

    print("\n" + "=" * 60)
    print("LINUX ASSISTANT RESULT")
    print("=" * 60)

    print(f"Query: {final_result.query}")
    print(f"Domains analyzed: {', '.join(final_result.domains)}")

    response_data = final_result.response  # This is now always a dict

    if final_result.response_type == "command":
        # Validate structure before accessing keys
        assert type(response_data) is CommandResponse
        command = response_data.command
        explanation = response_data.explanation
        security_notes = response_data.security_notes

        print("\nCOMMAND FOR YOUR SYSTEM:")
        print(f"$ {command}")
        print("\nEXPLANATION:")
        print(explanation)
        if security_notes:
            print("\nSECURITY NOTES:")
            print(security_notes)
    else:  # Information response
        # Validate structure before accessing keys
        assert type(response_data) is InformationResponse
        answer = response_data.answer
        sources = response_data.sources

        print("\nABOUT YOUR SYSTEM:")
        print(answer)
        if sources:
            print("\nSOURCES FROM YOUR SYSTEM:")
            # Ensure sources is a list
            assert isinstance(sources, list)
            for source in sources:
                print(f"- {source}")

    print("\n" + "=" * 60)

    # Record this interaction in conversation history
    try:
        # Create a conversation entry
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": state.get(
                "original_prompt", state["prompt"]
            ),  # Use original if available
            "refined_query": state["prompt"] if state.get("original_prompt") else None,
            "domains": final_result.domains,
            "response_type": final_result.response_type,
            "response": final_result.response,
        }

        # Initialize history if not present
        if "conversation_history" not in state:
            state["conversation_history"] = []

        # Add entry to history
        state["conversation_history"].append(entry)

        # Log the addition
        history_length = len(state["conversation_history"])
        print(f"Conversation history updated. Now contains {history_length} entries.")

    except Exception as e:
        print(f"Warning: Could not record conversation history: {e}")

    return state
