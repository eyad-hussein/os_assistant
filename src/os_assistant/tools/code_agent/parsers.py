from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser

from models import CodeAnalysis

def create_code_analysis_parser():
    """Create a parser for the CodeAnalysis model"""
    return PydanticOutputParser(pydantic_object=CodeAnalysis)

def get_parsing_instructions():
    """Get parsing instructions for the LLM"""
    parser = create_code_analysis_parser()
    return parser.get_format_instructions()

def parse_structured_output(response_text, model_class):
    """Parse structured output from LLM response"""
    parser = PydanticOutputParser(pydantic_object=model_class)
    return parser.parse(response_text)

def extract_code_from_markdown(text):
    """Extract code from markdown code blocks"""
    if "```python" in text and "```" in text:
        return text.split("```python")[1].split("```")[0].strip()
    elif "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    return text
