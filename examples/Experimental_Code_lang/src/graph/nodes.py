import json
from typing import Dict, List, Optional

from langchain.schema import HumanMessage

# Assuming rag_manager is initialized and available globally or passed appropriately
# For simplicity, let's assume it's imported from the root level
# In a larger app, dependency injection would be better.
from rag_manager import get_rag_manager

from ..config.settings import DOMAINS, model
from ..models.schemas import (
    CommandResponse,
    DomainAnalysis,
    FinalResult,
    InformationResponse,
    QueryTypeResult,
)
from ..parsers.setup import (
    command_response_parser,
    domain_analysis_parser,
    fixed_command_response_parser,
    fixed_domain_analysis_parser,
    fixed_info_response_parser,
    fixed_query_type_parser,
    info_response_parser,
    parse_with_fix_and_extract,  # Import the helper
    query_type_parser,
)
from .state import LinuxAssistantState

# Get RAG manager instance
rag_manager = get_rag_manager()

# --- Node Functions ---


def initialize_state(state: LinuxAssistantState, prompt: str) -> LinuxAssistantState:
    """Initialize the state with user prompt"""
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
    return state


def domain_analysis_node(state: LinuxAssistantState) -> LinuxAssistantState:
    """Analyze which domains are relevant to the query"""
    print("\nAnalyzing query domains...")

    # Strengthened prompt demanding ONLY JSON
    prompt = f"""Analyze this Linux query: '{state["prompt"]}' and identify the most relevant domains from: {', '.join(state["domains"])}.

    Guidelines:
    1. Consider that queries may relate to multiple domains.
    2. Focus on the primary intent of the query.
    3. Include only truly relevant domains.
    4. Provide reasoning.

    Domains overview:
    - filesystem: File operations, directories, permissions, storage
    - users: User accounts, passwords, authentication, user groups
    - packages: Software installation, updates, package management
    - network: Connectivity, IP configuration, networking tools

    IMPORTANT: Your response MUST be ONLY a valid JSON object conforming to the specified format.
    Do NOT include any introductory text, explanations, apologies, or any characters before the opening '{{' or after the closing '}}'.

    JSON Format:
    {domain_analysis_parser.get_format_instructions()}
    """

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

        state["domain_analysis"] = domain_analysis.model_dump()
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
        state["domain_analysis"] = fallback_analysis.model_dump()
        state["domains_to_process"] = state[
            "domains"
        ].copy()  # Use all available domains

    return state


def context_retrieval_node(state: LinuxAssistantState) -> LinuxAssistantState:
    """Retrieve context for a domain"""
    if not state["domains_to_process"]:
        print("No more domains to process for context retrieval.")
        return state  # No more domains to process

    current_domain = state["domains_to_process"].pop(0)
    state["current_domain"] = current_domain

    print(f"\nRetrieving context for domain: {current_domain}")

    try:
        # Get context from RAG manager
        # Ensure rag_manager is accessible here
        context = rag_manager.retrieve_formatted(current_domain, state["prompt"], 3)

        # Store the context
        state["contexts"][current_domain] = context
        print(f"Retrieved context from {current_domain}")

    except Exception as e:
        print(f"Error retrieving context for {current_domain}: {str(e)}")
        state["contexts"][current_domain] = (
            f"Error retrieving context for {current_domain}"
        )

    # Clear current_domain after processing
    state["current_domain"] = None
    return state


def query_classifier_node(state: LinuxAssistantState) -> LinuxAssistantState:
    """Classify the query type (command or information)"""
    print("\nClassifying query type...")

    combined_context = ""
    # Use only contexts from the domains identified in the analysis step
    relevant_domains = (
        state["domain_analysis"]["domains"]
        if state["domain_analysis"]
        else state["domains"]
    )
    for domain in relevant_domains:
        context = state["contexts"].get(domain, "No context retrieved.")
        combined_context += f"--- {domain.upper()} DOMAIN ---\n{context}\n\n"

    if not combined_context:
        combined_context = "No specific context was retrieved for the relevant domains."

    # Strengthened prompt demanding ONLY JSON
    prompt = f"""Classify this query: '{state["prompt"]}' as either 'command' or 'information'.

    Context from relevant domains:
    {combined_context}

    Guidelines:
    - 'command': User wants to perform an action or needs a Linux command.
    - 'information': User wants facts, explanations, or understanding.

    Provide reasoning for the classification.

    IMPORTANT: Your response MUST be ONLY a valid JSON object conforming to the specified format.
    Do NOT include any introductory text, explanations, apologies, or any characters before the opening '{{' or after the closing '}}'.

    JSON Format:
    {query_type_parser.get_format_instructions()}
    """

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

        state["query_type"] = query_type.model_dump()

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
        state["query_type"] = fallback_query_type.model_dump()

    return state


def command_generator_node(state: LinuxAssistantState) -> LinuxAssistantState:
    """Generate a command response"""
    print("\nGenerating Linux command...")

    combined_context = ""
    # Use only contexts from the domains identified in the analysis step
    relevant_domains = (
        state["domain_analysis"]["domains"]
        if state["domain_analysis"]
        else state["domains"]
    )
    for domain in relevant_domains:
        context = state["contexts"].get(domain, "No context retrieved.")
        combined_context += f"--- {domain.upper()} DOMAIN ---\n{context}\n\n"

    if not combined_context:
        combined_context = "No specific context was retrieved for the relevant domains."

    # Strengthened prompt demanding ONLY JSON
    prompt = f"""Generate a Linux command for: '{state["prompt"]}' based on these domains: {', '.join(relevant_domains)} and the following context:
    {combined_context}

    IMPORTANT: The context contains information from the user's actual system. Tailor the command to their environment based on the context.

    Response Requirements:
    1. A single, executable Linux command.
    2. A brief explanation specific to the user's system.
    3. Any relevant security considerations.

    Use specific details (paths, usernames) from the context. Refer to the user's environment directly (e.g., "your system").

    IMPORTANT: Your response MUST be ONLY a valid JSON object conforming to the specified format.
    Do NOT include any introductory text, explanations, apologies, or any characters before the opening '{{' or after the closing '}}'.

    JSON Format:
    {command_response_parser.get_format_instructions()}
    """

    messages = [HumanMessage(content=prompt)]
    content = model.invoke(messages)

    try:
        # Use the helper function for parsing attempts
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

        state["command_response"] = command_response.model_dump()

        print(f"Generated command: {command_response.command}")

    except Exception as e:
        print(f"Error generating command: {str(e)}")
        # Fallback command
        fallback_command = CommandResponse(
            command="echo 'Could not generate a specific command for your request'",
            explanation=f"I was unable to generate a precise command for '{state['prompt']}' based on your system context.",
            security_notes="Please review any command carefully before execution.",
        )
        state["command_response"] = fallback_command.model_dump()

    return state


def information_generator_node(state: LinuxAssistantState) -> LinuxAssistantState:
    """Generate an information response"""
    print("\nGenerating information response...")

    combined_context = ""
    # Use only contexts from the domains identified in the analysis step
    relevant_domains = (
        state["domain_analysis"]["domains"]
        if state["domain_analysis"]
        else state["domains"]
    )
    for domain in relevant_domains:
        context = state["contexts"].get(domain, "No context retrieved.")
        combined_context += f"--- {domain.upper()} DOMAIN ---\n{context}\n\n"

    if not combined_context:
        combined_context = "No specific context was retrieved for the relevant domains."

    # Strengthened prompt demanding ONLY JSON
    prompt = f"""Answer this question: '{state["prompt"]}' using the following context:
    {combined_context}

    IMPORTANT INSTRUCTIONS:
    - The context contains information from the user's ACTUAL SYSTEM.
    - Treat this as personalized information about THEIR specific device/environment.
    - Reference specific details from the context (timestamps, paths, usernames, etc.).
    - Use phrases like "your system", "on your device", "in your configuration".

    IMPORTANT: Your response MUST be ONLY a valid JSON object conforming to the specified format.
    Do NOT include any introductory text, explanations, apologies, or any characters before the opening '{{' or after the closing '}}'.

    JSON Format:
    {info_response_parser.get_format_instructions()}
    """

    messages = [HumanMessage(content=prompt)]
    content = model.invoke(messages)

    try:
        # Use the helper function for parsing attempts
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

        state["information_response"] = info_response.model_dump()

        print("Successfully generated information response")

    except Exception as e:
        print(f"Error in information generation: {str(e)}")

        # Check if the query was too vague or generic
        if state["prompt"].lower() in ["bla", "test", "hi", "hello"]:
            answer = f"Your query '{state['prompt']}' is too short or generic. For better results, please ask a specific question about your Linux system."
        else:
            answer = f"I'm having trouble finding specific information about '{state['prompt']}' on your system. Could you provide more details or try a different query?"

        # Fallback information response
        fallback_info = InformationResponse(answer=answer, sources=["System analysis"])
        state["information_response"] = fallback_info.model_dump()

    return state


def prepare_final_result_node(state: LinuxAssistantState) -> LinuxAssistantState:
    """Prepare the final result"""
    # Ensure domain_analysis and query_type exist before accessing keys
    if not state.get("domain_analysis"):
        print("Warning: Domain analysis missing, using all domains for final result.")
        domains = state["domains"]  # Fallback to all domains
    else:
        domains = state["domain_analysis"]["domains"]

    if not state.get("query_type"):
        print(
            "Warning: Query type missing, defaulting to 'information' for final result."
        )
        response_type = "information"  # Fallback type
    else:
        response_type = state["query_type"]["query_type"]

    # Create context summary
    context_summary = "Analyzed information from: "
    context_summary += ", ".join(domains)

    # Determine response content
    response: Optional[Dict] = None
    if response_type == "command":
        if state.get("command_response"):
            response = state["command_response"]
        else:
            print("Warning: Command response expected but missing.")
            # Create a fallback command response if needed, or switch type
            response_type = "information"  # Switch to info if command failed
            response = InformationResponse(
                answer=f"Could not generate a command for '{state['prompt']}'. Please try rephrasing.",
                sources=["System processing error"],
            ).model_dump()

    # Handle information response (either primary or fallback)
    if response_type == "information":
        if state.get("information_response"):
            response = state["information_response"]
        else:
            print("Warning: Information response expected but missing.")
            # Create a fallback information response
            response = InformationResponse(
                answer=f"Unable to generate an answer for '{state['prompt']}' based on the available information.",
                sources=["System processing error"],
            ).model_dump()

    # Ensure response is not None before creating FinalResult
    if response is None:
        print("Error: Could not determine a valid response for the final result.")
        # Handle this case, maybe set final_result to an error state or raise exception
        # For now, create a minimal error response
        response = InformationResponse(
            answer="An unexpected error occurred while generating the response.",
            sources=["System error"],
        ).model_dump()
        response_type = "information"  # Ensure type matches the fallback

    # Create final result
    final_result = FinalResult(
        query=state["prompt"],
        domains=domains,
        response_type=response_type,  # Use the potentially updated response_type
        response=response,  # Pass the dictionary directly
        context_summary=context_summary,
    )

    state["final_result"] = final_result.model_dump()

    return state


def display_result_node(state: LinuxAssistantState) -> LinuxAssistantState:
    """Display the final result to the user"""
    if not state.get("final_result"):
        print("\nError: No final result generated.")
        return state

    final_result = state["final_result"]

    print("\n" + "=" * 60)
    print("LINUX ASSISTANT RESULT")
    print("=" * 60)

    print(f"Query: {final_result['query']}")
    print(f"Domains analyzed: {', '.join(final_result['domains'])}")

    response_data = final_result["response"]  # This is now always a dict

    if final_result["response_type"] == "command":
        # Validate structure before accessing keys
        command = response_data.get("command", "N/A")
        explanation = response_data.get("explanation", "No explanation provided.")
        security_notes = response_data.get("security_notes")

        print("\nCOMMAND FOR YOUR SYSTEM:")
        print(f"$ {command}")
        print("\nEXPLANATION:")
        print(explanation)
        if security_notes:
            print("\nSECURITY NOTES:")
            print(security_notes)
    else:  # Information response
        # Validate structure before accessing keys
        answer = response_data.get("answer", "No answer provided.")
        sources = response_data.get("sources")

        print("\nABOUT YOUR SYSTEM:")
        print(answer)
        if sources:
            print("\nSOURCES FROM YOUR SYSTEM:")
            # Ensure sources is a list
            if isinstance(sources, list):
                for source in sources:
                    print(f"- {source}")
            else:
                print(f"- {sources}")  # Handle if sources is not a list unexpectedly

    print("\n" + "=" * 60)
    print("===", "here is self.state:", f"{state=}")
    return state
