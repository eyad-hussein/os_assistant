"""
Query Router for Hybrid RAG + SQL Retrieval.

Determines whether a user query should use:
1. Structured SQL queries (via MCP) - for precise, structured data
2. Semantic RAG search - for understanding and summarization
3. Both (hybrid) - for complex queries needing both precision and context
"""

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from langchain.schema import HumanMessage

from os_assistant.utils import LOGGER
from os_assistant.utils.model_factory import model


class QueryIntent(Enum):
    """Classification of query intent for routing."""

    STRUCTURED = "structured"  # Use SQL for precise queries
    SEMANTIC = "semantic"  # Use RAG for understanding
    HYBRID = "hybrid"  # Use both SQL + RAG


@dataclass
class RouterDecision:
    """Result of query routing decision."""

    intent: QueryIntent
    sql_query: str | None = None
    rag_query: str | None = None
    confidence: float = 0.0
    reasoning: str = ""


ROUTER_PROMPT = """You are a query router for an OS monitoring system. 
Analyze the user's query and determine the best retrieval strategy.

DATABASE SCHEMA:
{schema}

USER QUERY: {query}

Determine if this query needs:
1. STRUCTURED - Precise SQL query (dates, specific files, counts, filters, listing)
2. SEMANTIC - Understanding/summarization (why, how, explain, help with, troubleshooting)
3. HYBRID - Both (complex questions needing both precise data AND context)

IMPORTANT SQL GUIDELINES:
- Use the table name from the schema (e.g., 'file_system' for filesystem logs)
- For time-based queries, use SQLite datetime functions: datetime('now'), datetime('now', '-1 day'), etc.
- Common event values: 'created', 'modified', 'deleted', 'moved'
- The column for event type is 'event' (not 'event_type')
- The column for file path is 'full_path' (not 'source_path')
- Always include ORDER BY for time-based queries (usually DESC for recent first)
- Use LIMIT to avoid returning too many rows (default LIMIT 20)

Respond in this EXACT JSON format:
{{
    "intent": "STRUCTURED" | "SEMANTIC" | "HYBRID",
    "sql_query": "SQL query if STRUCTURED or HYBRID, otherwise null",
    "rag_query": "Search query for RAG if SEMANTIC or HYBRID, otherwise null",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation"
}}

Examples:
- "What files were deleted yesterday?" → STRUCTURED (specific time filter)
  SQL: SELECT * FROM file_system WHERE event = 'deleted' AND timestamp >= datetime('now', '-1 day') ORDER BY timestamp DESC LIMIT 20
  
- "Why is my system slow?" → SEMANTIC (needs understanding)
  RAG: system performance slow issues

- "Show deleted files and explain what might have caused it" → HYBRID
  SQL: SELECT * FROM file_system WHERE event = 'deleted' ORDER BY timestamp DESC LIMIT 10
  RAG: file deletion causes reasons analysis

- "List recent file changes" → STRUCTURED
  SQL: SELECT * FROM file_system ORDER BY timestamp DESC LIMIT 20

RESPOND WITH JSON ONLY, NO OTHER TEXT:"""


class QueryRouter:
    """
    Routes queries to appropriate retrieval mechanism.

    Uses an LLM to analyze queries and determine whether they need:
    - Structured SQL queries (precise data)
    - Semantic RAG search (understanding)
    - Hybrid approach (both)
    """

    def __init__(self, schema: str = None):
        """
        Initialize the query router.

        Args:
            schema: Database schema string for SQL generation
        """
        self._schema = schema or self._get_default_schema()
        self._model = model

    def _get_default_schema(self) -> str:
        """Get default schema if none provided."""
        return """Table: file_system
Columns:
- id: INTEGER (Primary Key)
- event: VARCHAR (values: 'created', 'modified', 'deleted', 'moved')
- name: VARCHAR (file or directory name)
- is_directory: BOOLEAN
- full_path: VARCHAR (complete file path)
- timestamp: DATETIME"""

    def update_schema(self, schema: str):
        """Update the schema used for SQL generation."""
        self._schema = schema

    def route(self, query: str) -> RouterDecision:
        """
        Route a query to the appropriate retrieval mechanism.

        Args:
            query: User's natural language query

        Returns:
            RouterDecision with intent and generated queries
        """
        LOGGER.info(f"Routing query: {query[:50]}...")

        try:
            # Build the prompt
            prompt = ROUTER_PROMPT.format(schema=self._schema, query=query)

            # Get LLM response
            response = self._model.invoke([HumanMessage(content=prompt)])
            response_text = response.content.strip()

            # Parse the response
            decision = self._parse_response(response_text, query)
            LOGGER.info(
                f"Router decision: {decision.intent.value} (confidence: {decision.confidence})"
            )
            return decision

        except Exception as e:
            LOGGER.error(f"Query routing error: {e}")
            # Default to semantic search on error
            return RouterDecision(
                intent=QueryIntent.SEMANTIC,
                rag_query=query,
                confidence=0.5,
                reasoning=f"Fallback to semantic search due to error: {str(e)}",
            )

    def _parse_response(
        self, response_text: str, original_query: str
    ) -> RouterDecision:
        """
        Parse the LLM response into a RouterDecision.

        Args:
            response_text: Raw LLM response
            original_query: Original user query for fallback

        Returns:
            Parsed RouterDecision
        """
        try:
            # Try to extract JSON from the response
            # Handle cases where LLM might add markdown code blocks
            json_text = response_text
            if "```json" in response_text:
                match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
                if match:
                    json_text = match.group(1)
            elif "```" in response_text:
                match = re.search(r"```\s*(.*?)\s*```", response_text, re.DOTALL)
                if match:
                    json_text = match.group(1)

            # Parse JSON
            data = json.loads(json_text)

            # Map intent string to enum
            intent_str = data.get("intent", "SEMANTIC").upper()
            intent_map = {
                "STRUCTURED": QueryIntent.STRUCTURED,
                "SEMANTIC": QueryIntent.SEMANTIC,
                "HYBRID": QueryIntent.HYBRID,
            }
            intent = intent_map.get(intent_str, QueryIntent.SEMANTIC)

            return RouterDecision(
                intent=intent,
                sql_query=data.get("sql_query"),
                rag_query=data.get("rag_query"),
                confidence=float(data.get("confidence", 0.7)),
                reasoning=data.get("reasoning", ""),
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            LOGGER.warning(f"Failed to parse router response: {e}")
            # Fallback: try to detect intent from keywords
            return self._fallback_routing(original_query)

    def _fallback_routing(self, query: str) -> RouterDecision:
        """
        Fallback routing based on keyword detection.

        Args:
            query: User's query

        Returns:
            RouterDecision based on keyword analysis
        """
        query_lower = query.lower()

        # Keywords suggesting structured SQL queries
        sql_keywords = [
            "list",
            "show",
            "count",
            "how many",
            "yesterday",
            "today",
            "last",
            "recent",
            "deleted",
            "created",
            "modified",
            "files",
            "folders",
            "directories",
        ]

        # Keywords suggesting semantic understanding
        semantic_keywords = [
            "why",
            "how",
            "explain",
            "help",
            "understand",
            "troubleshoot",
            "fix",
            "solve",
            "analyze",
        ]

        sql_score = sum(1 for kw in sql_keywords if kw in query_lower)
        semantic_score = sum(1 for kw in semantic_keywords if kw in query_lower)

        if sql_score > semantic_score:
            return RouterDecision(
                intent=QueryIntent.STRUCTURED,
                sql_query=self._generate_fallback_sql(query),
                confidence=0.6,
                reasoning="Fallback: detected data-oriented keywords",
            )
        elif semantic_score > sql_score:
            return RouterDecision(
                intent=QueryIntent.SEMANTIC,
                rag_query=query,
                confidence=0.6,
                reasoning="Fallback: detected understanding-oriented keywords",
            )
        else:
            # Default to hybrid for ambiguous queries
            return RouterDecision(
                intent=QueryIntent.HYBRID,
                sql_query=self._generate_fallback_sql(query),
                rag_query=query,
                confidence=0.5,
                reasoning="Fallback: ambiguous query, using hybrid approach",
            )

    def _generate_fallback_sql(self, query: str) -> str:
        """
        Generate a simple fallback SQL query.

        Args:
            query: User's query

        Returns:
            Basic SQL query string
        """
        query_lower = query.lower()

        # Detect event type
        event_type = None
        if "deleted" in query_lower or "delete" in query_lower:
            event_type = "deleted"
        elif (
            "created" in query_lower or "create" in query_lower or "new" in query_lower
        ):
            event_type = "created"
        elif (
            "modified" in query_lower
            or "changed" in query_lower
            or "edited" in query_lower
        ):
            event_type = "modified"
        elif "moved" in query_lower or "renamed" in query_lower:
            event_type = "moved"

        # Build query
        base_query = "SELECT * FROM file_system"
        conditions = []

        if event_type:
            conditions.append(f"event = '{event_type}'")

        # Time-based filtering
        if "yesterday" in query_lower:
            conditions.append("timestamp >= datetime('now', '-1 day')")
        elif "today" in query_lower:
            conditions.append("timestamp >= datetime('now', 'start of day')")
        elif "last hour" in query_lower:
            conditions.append("timestamp >= datetime('now', '-1 hour')")
        elif "last week" in query_lower:
            conditions.append("timestamp >= datetime('now', '-7 days')")

        if conditions:
            base_query += " WHERE " + " AND ".join(conditions)

        base_query += " ORDER BY timestamp DESC LIMIT 20"

        return base_query


# Singleton router instance
_router: QueryRouter | None = None


def get_query_router(schema: str = None) -> QueryRouter:
    """
    Get or create singleton QueryRouter instance.

    Args:
        schema: Optional schema to initialize/update the router with

    Returns:
        The QueryRouter instance
    """
    global _router
    if _router is None:
        _router = QueryRouter(schema)
    elif schema:
        _router.update_schema(schema)
    return _router
