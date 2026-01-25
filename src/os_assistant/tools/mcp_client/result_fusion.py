import threading
from dataclasses import dataclass, field
from typing import Any

from os_assistant.utils import LOGGER


@dataclass
class FusedResult:
    """Combined result from SQL and RAG retrieval."""

    sql_context: str = ""
    rag_context: str = ""
    combined_context: str = ""
    sources: list[str] = field(default_factory=list)
    sql_row_count: int = 0
    rag_doc_count: int = 0
    has_sql: bool = False
    has_rag: bool = False


class ResultFusion:
    """
    Fuses results from SQL and RAG retrieval into unified context.

    This class combines structured data from SQL queries with semantic
    understanding from RAG to provide comprehensive context for the LLM.
    """

    def __init__(self):
        """Initialize the result fusion engine."""
        pass

    def fuse(
        self,
        sql_result: Any = None,
        rag_result: str = None,
        query: str = "",
    ) -> FusedResult:
        """
        Fuse SQL and RAG results into a unified context.

        Args:
            sql_result: Result from SQL query (MCPToolResult or dict)
            rag_result: Result from RAG search (string context)
            query: Original user query for context

        Returns:
            FusedResult with combined context
        """
        result = FusedResult()

        # Process SQL result
        if sql_result is not None:
            result.sql_context, result.sql_row_count = self._format_sql_result(
                sql_result
            )
            result.has_sql = bool(result.sql_context)
            if result.has_sql:
                result.sources.append("SQL Database")

        # Process RAG result
        if rag_result is not None and rag_result.strip():
            result.rag_context = rag_result
            result.has_rag = True
            result.sources.append("RAG Semantic Search")
            # Estimate document count from context
            result.rag_doc_count = result.rag_context.count("Log #")

        # Build combined context
        result.combined_context = self._build_combined_context(result, query)

        LOGGER.info(
            f"Fused results: SQL={result.sql_row_count} rows, RAG={result.rag_doc_count} docs"
        )

        return result

    def _format_sql_result(self, sql_result: Any) -> tuple[str, int]:
        """
        Format SQL result into readable context.

        Args:
            sql_result: Raw SQL result (MCPToolResult or dict/list)

        Returns:
            Tuple of (formatted string, row count)
        """
        try:
            # Handle MCPToolResult
            if hasattr(sql_result, "success"):
                if not sql_result.success:
                    return f"SQL Query Error: {sql_result.error}", 0

                data = sql_result.data
            else:
                data = sql_result

            # Handle different data formats
            if isinstance(data, list):
                if not data:
                    return "No results found in database.", 0

                # Format as a readable table
                formatted_rows = []
                for i, row in enumerate(data, 1):
                    if isinstance(row, dict):
                        row_str = self._format_row(row, i)
                        formatted_rows.append(row_str)
                    else:
                        formatted_rows.append(f"Row {i}: {row}")

                row_count = len(data)
                header = f"=== DATABASE RESULTS ({row_count} rows) ===\n"
                return header + "\n".join(formatted_rows), row_count

            elif isinstance(data, dict):
                if "data" in data:
                    return self._format_sql_result(data["data"])
                elif "message" in data:
                    return data["message"], 0
                else:
                    return str(data), 1

            else:
                return str(data), 1

        except Exception as e:
            LOGGER.error(f"Error formatting SQL result: {e}")
            return f"Error processing SQL result: {e}", 0

    def _format_row(self, row: dict, index: int) -> str:
        """
        Format a single database row.

        Args:
            row: Dictionary representing a database row
            index: Row index for display

        Returns:
            Formatted row string
        """
        parts = [f"\n--- Entry {index} ---"]

        # Priority fields to show first
        priority_fields = [
            "timestamp",
            "event",
            "name",
            "full_path",
            "is_directory",
        ]

        # Show priority fields first
        for field in priority_fields:
            if field in row and row[field] is not None:
                value = row[field]
                # Format the field name nicely
                field_name = field.replace("_", " ").title()
                parts.append(f"  {field_name}: {value}")

        # Show remaining fields
        for key, value in row.items():
            if key not in priority_fields and key != "id" and value is not None:
                field_name = key.replace("_", " ").title()
                parts.append(f"  {field_name}: {value}")

        return "\n".join(parts)

    def _build_combined_context(self, result: FusedResult, query: str) -> str:
        """
        Build the final combined context string.

        Args:
            result: FusedResult with SQL and RAG contexts
            query: Original user query

        Returns:
            Combined context string for LLM
        """
        sections = []

        # Header with sources
        sources_str = " & ".join(result.sources) if result.sources else "No sources"
        sections.append(f"=== RETRIEVED CONTEXT (Sources: {sources_str}) ===\n")

        # Add SQL context if available
        if result.has_sql and result.sql_context:
            sections.append("--- STRUCTURED DATA (from Database) ---")
            sections.append(result.sql_context)
            sections.append("")

        # Add RAG context if available
        if result.has_rag and result.rag_context:
            sections.append("--- SEMANTIC CONTEXT (from Log Analysis) ---")
            sections.append(result.rag_context)
            sections.append("")

        # Add summary
        if result.has_sql and result.has_rag:
            sections.append("--- CONTEXT SUMMARY ---")
            sections.append(
                f"Retrieved {result.sql_row_count} database records and "
                f"{result.rag_doc_count} log documents for comprehensive analysis."
            )
        elif not result.has_sql and not result.has_rag:
            sections.append("No relevant context was retrieved for this query.")

        return "\n".join(sections)


# Singleton instance
_fusion: ResultFusion | None = None
_fusion_lock = threading.Lock()


def get_result_fusion() -> ResultFusion:
    """
    Get or create singleton ResultFusion instance (thread-safe).

    Returns:
        The ResultFusion instance
    """
    global _fusion
    if _fusion is None:
        with _fusion_lock:
            if _fusion is None:
                _fusion = ResultFusion()
    return _fusion
