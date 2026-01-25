from langchain.schema import HumanMessage

from os_assistant.core.nodes.helpers import is_rag_enabled
from os_assistant.core.state import AssistantState
from os_assistant.parsers.setup import (
    domain_analysis_parser,
    fixed_domain_analysis_parser,
    parse_with_fix_and_extract,
)
from os_assistant.prompts.prompt_loader import load_prompt
from os_assistant.pydantic_models.schemas import DomainAnalysis
from os_assistant.utils import LOGGER
from os_assistant.utils.model_factory import model


def domain_analysis_node(state: AssistantState) -> AssistantState:
    """Analyze which domains are relevant to the query"""
    LOGGER.info("\nNODE: domain_analysis_node")
    LOGGER.info("\nAnalyzing query domains...")

    if not is_rag_enabled():
        LOGGER.warning("RAG disabled in current mode. No domains will be used.")
        fallback_analysis = DomainAnalysis(
            domains=[],
            confidence=1.0,
            reasoning="No domains needed as RAG is disabled in the current mode.",
            requires_logs=False,
        )
        state["domain_analysis"] = fallback_analysis
        state["domains_to_process"] = []
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
        # Only process domains if logs are required
        state["domains_to_process"] = (
            domain_analysis.domains.copy() if domain_analysis.requires_logs else []
        )

        LOGGER.info(f"Domains identified: {domain_analysis.domains}")
        LOGGER.info(f"Requires logs: {domain_analysis.requires_logs}")
        LOGGER.info(f"Domains to process: {state['domains_to_process']}")
        LOGGER.info(f"Confidence: {domain_analysis.confidence}")
        LOGGER.info(f"Reasoning: {domain_analysis.reasoning}")

    except Exception as e:
        LOGGER.error(f"Domain analysis error: {str(e)}")
        # Fallback to no domains - domain analysis is for log retrieval
        fallback_analysis = DomainAnalysis(
            domains=[],
            confidence=0.5,
            reasoning=f"No domains selected due to error: {str(e)}",
            requires_logs=False,
        )
        state["domain_analysis"] = fallback_analysis
        state["domains_to_process"] = []

    return state
