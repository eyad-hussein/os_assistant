"""Tools package for OS Assistant.

Contains:
- agentic_rag: RAG-based log search
- code_agent: Code execution agent
- mcp_client: MCP client for tracer integration
- vision: Multimodal vision analysis
"""

# MCP client exports
from .mcp_client import MCPClientWrapper, QueryIntent, QueryRouter

# Vision module exports
from .vision import VisionAnalyzer, VisionConfig, get_vision_analyzer

__all__ = [
    # Vision
    "VisionAnalyzer",
    "VisionConfig",
    "get_vision_analyzer",
    # MCP
    "MCPClientWrapper",
    "QueryRouter",
    "QueryIntent",
]
