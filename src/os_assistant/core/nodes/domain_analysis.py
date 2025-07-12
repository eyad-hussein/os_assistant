from os_assistant.core.state import AssistantState
from os_assistant.core.nodes.helpers import is_rag_enabled
from os_assistant.utils.settings import ASSISTANT_MODE
from os_assistant.pydantic_models.schemas import DomainAnalysis
from os_assistant.prompts.prompt_loader import load_prompt
from langchain.schema import HumanMessage
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

    if not is_rag_enabled():
        print("RAG disabled in current mode. Using all domains.")
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

    try:
        # Load and format the domain analysis prompt
        domain_analysis_yaml = load_prompt("domain_analysis_node")
        prompt = domain_analysis_yaml["prompt"].format(
            prompt=state["prompt"],
            domains=", ".join(state["domains"]),
            format_instructions=domain_analysis_parser.get_format_instructions(),
        )
        messages = [HumanMessage(content=prompt)]
        content = model.invoke(messages)

        domain_analysis = parse_with_fix_and_extract(
            content,
            domain_analysis_parser,
            fixed_domain_analysis_parser,
        )

        # Ensure the result is a DomainAnalysis object
        if not isinstance(domain_analysis, DomainAnalysis):
            domain_analysis = DomainAnalysis.model_validate(domain_analysis)

        state["domain_analysis"] = domain_analysis
        state["domains_to_process"] = domain_analysis.domains.copy()

        print(f"Domains identified: {domain_analysis.domains}")
        print(f"Confidence: {domain_analysis.confidence}")
        print(f"Reasoning: {domain_analysis.reasoning}")

    except Exception as e:
        print(f"Domain analysis error: {str(e)}")
        # Fallback to all domains in case of any errors
        fallback_analysis = DomainAnalysis(
            domains=state["domains"],
            confidence=0.5,
            reasoning=f"Fallback to all domains due to error: {str(e)}",
        )
        state["domain_analysis"] = fallback_analysis
        state["domains_to_process"] = state["domains"].copy()

    return state