from os_assistant.core.state import AssistantState
from os_assistant.core.nodes.helpers import is_rag_enabled
from os_assistant.utils.settings import ASSISTANT_MODE
from os_assistant.pydantic_models.schemas import DomainAnalysis
from os_assistant.prompts.prompt_loader import load_prompt
from langchain.schema import HumanMessage, SystemMessage
from os_assistant.utils.model_factory import model
from os_assistant.parsers.setup import (
    domain_analysis_parser,
    fixed_domain_analysis_parser,
    parse_with_fix_and_extract,
)

def domain_analysis_node(state: AssistantState) -> AssistantState:
    """Analyze which domains are relevant to the query"""
    print("\nNODE: domain_analysis_node")
    print("\nAnalyzing query domains...")

    # Check if RAG is enabled
    if not is_rag_enabled():
        print("RAG disabled in current mode. Using all domains.")
        # Create a simple domain analysis without RAG
        fallback_analysis = DomainAnalysis(
            domains=state["domains"],
            confidence=0.5,
            reasoning="Using all available domains as RAG is disabled in the current mode.",
        )
        state["domain_analysis"] = fallback_analysis
        state["domains_to_process"] = (
            state["domains"].copy() if ASSISTANT_MODE == 1 else []
        )
        # For tool-only mode, we still want to collect domains but will skip context retrieval
        return state

    # RAG is enabled, continue with normal domain analysis
    try:
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

            # Ensure the result is a Pydantic model instance
            if not isinstance(domain_analysis, DomainAnalysis):
                domain_analysis = DomainAnalysis.model_validate(domain_analysis)

            state["domain_analysis"] = domain_analysis
            state["domains_to_process"] = domain_analysis.domains.copy()

            print(f"Domains identified: {domain_analysis.domains}")
            print(f"Confidence: {domain_analysis.confidence}")
            print(f"Reasoning: {domain_analysis.reasoning}")

        except Exception as e:
            print(f"Error analyzing domains: {str(e)}")
            # Fallback to using all domains
            fallback_analysis = DomainAnalysis(
                domains=state["domains"],
                confidence=0.5,
                reasoning=f"Fallback: using all available domains due to analysis error for query: '{state['prompt']}'",
            )
            state["domain_analysis"] = fallback_analysis
            state["domains_to_process"] = state["domains"].copy()
    except Exception as e:
        print(f"Critical error in domain analysis: {str(e)}")
        # Ensure we always have a valid domain analysis even if everything fails
        fallback_analysis = DomainAnalysis(
            domains=["file_system"],  # Default to file_system as the safest fallback
            confidence=0.1,
            reasoning=f"Emergency fallback due to critical error: {str(e)}",
        )
        state["domain_analysis"] = fallback_analysis
        state["domains_to_process"] = ["file_system"]

    return state