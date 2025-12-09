from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import PromptTemplate

from ....utils.model_factory import fixing_model
from ..core.models import CodeAnalysis
from .json_parsers import extract_json_manually
from .markdown_parsers import extract_code_from_markdown


def create_code_analysis_parser() -> PydanticOutputParser[CodeAnalysis]:
    """Create a parser for the CodeAnalysis model."""
    return PydanticOutputParser(pydantic_object=CodeAnalysis)


def create_output_fixing_prompt():
    """Create a prompt template for fixing malformed JSON."""
    template = """
    Instructions:
    The following output was intended to be valid JSON conforming to the schema below, but it is malformed.
    Please extract the valid JSON object from the output. Respond with ONLY the JSON object, nothing else.

    Schema:
    {instructions}

    Malformed Output:
    {completion}

    Error Details:
    {error}

    Corrected JSON Output:
    """
    return PromptTemplate.from_template(template)


def create_fixing_parser(parser):
    """Create an OutputFixingParser with increased retries."""
    llm = fixing_model
    prompt = create_output_fixing_prompt()
    return OutputFixingParser.from_llm(
        parser=parser, llm=llm, prompt=prompt, max_retries=5
    )


def parse_structured_output(response_text, model_class):
    """Parse structured output from LLM response with multiple fallback mechanisms."""
    parser = PydanticOutputParser(pydantic_object=model_class)
    fixing_parser = create_fixing_parser(parser)

    try:
        return parser.parse(response_text)
    except OutputParserException:
        try:
            return fixing_parser.parse(response_text)
        except OutputParserException:
            json_data = extract_json_manually(response_text)
            if json_data and "code" in json_data:
                return CodeAnalysis(
                    code=json_data.get("code", ""),
                    dangerous=json_data.get("dangerous", 1),
                    reason=json_data.get("reason", "Extracted manually from response"),
                )
            return CodeAnalysis(
                code=extract_code_from_markdown(response_text),
                dangerous=1,
                reason="Parser couldn't extract danger assessment, using default safe level.",
            )
