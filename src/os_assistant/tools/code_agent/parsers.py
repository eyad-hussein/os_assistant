from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
import json
import re

from .models import CodeAnalysis

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
    
    # First try to parse the response as is
    try:
        return parser.parse(response_text)
    except Exception:
        # Try multiple approaches to extract proper JSON
        
        # Try to extract JSON string using regex
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            try:
                json_str = json_match.group(1).strip()
                return parser.parse(json_str)
            except:
                pass
                
        # Try to extract based on curly braces
        try:
            json_str = extract_json_from_text(response_text)
            if json_str:
                return parser.parse(json_str)
        except:
            pass
            
        # If we find a dictionary-like pattern with the expected keys, try to clean it up
        if all(key in response_text for key in ["code", "dangerous", "reason"]):
            try:
                # Extract clean JSON using regex for each field
                code_match = re.search(r'"code"\s*:\s*(?:"""|\"{3})([\s\S]*?)(?:"""|\"{3})', response_text)
                dangerous_match = re.search(r'"dangerous"\s*:\s*(\d+)', response_text)
                reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', response_text)
                
                if code_match and dangerous_match:
                    code = code_match.group(1)
                    dangerous = int(dangerous_match.group(1))
                    reason = reason_match.group(1) if reason_match else "Extracted from response"
                    
                    return CodeAnalysis(
                        code=code,
                        dangerous=dangerous,
                        reason=reason
                    )
            except:
                pass
        
        # Final fallback: extract code and create a basic analysis
        code = extract_code_from_markdown(response_text)
        return CodeAnalysis(
            code=code,
            dangerous=1,
            reason="Parser couldn't extract danger assessment, using default safe level."
        )

def extract_json_from_text(text):
    """Extract JSON from text by finding sections between curly braces"""
    start = text.find('{')
    if start == -1:
        return None
    
    # Find matching closing brace
    open_count = 0
    for i in start, len(text):
        if text[i] == '{':
            open_count += 1
        elif text[i] == '}':
            open_count -= 1
            if open_count == 0:
                return text[start:i+1]
    
    return None

def extract_code_from_markdown(text):
    """Extract code from markdown code blocks"""
    if "```python" in text and "```" in text:
        return text.split("```python")[1].split("```")[0].strip()
    elif "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    return text
