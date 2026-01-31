import asyncio
import json
import threading
from dataclasses import dataclass, field
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from dagent.utils import LOGGER
from dagent.utils.settings import MCP_ENABLED, MCP_SERVER_URL


@dataclass
class MCPToolResult:
    """Result from an MCP tool call."""

    success: bool
    data: Any = None
    message: str = ""
    error: str | None = None


@dataclass
class MCPSchema:
    """Database schema information from MCP server."""

    domain: str
    schema_text: str
    tables: list[str] = field(default_factory=list)


class MCPClientWrapper:
    """
    Synchronous wrapper for MCP client operations.

    This class provides sync methods that wrap the async MCP client,
    making it compatible with LangGraph's synchronous node execution.
    """

    def __init__(self, server_url: str = None):
        """
        Initialize the MCP client wrapper.

        Args:
            server_url: URL of the MCP server. Defaults to settings.MCP_SERVER_URL
        """
        self.server_url = server_url or MCP_SERVER_URL
        self._schema_cache: dict[str, MCPSchema] = {}
        self._connected = False

    def is_enabled(self) -> bool:
        """Check if MCP is enabled in settings."""
        return MCP_ENABLED

    async def _execute_tool_async(
        self, tool_name: str, arguments: dict[str, Any], timeout: float = 30.0
    ) -> MCPToolResult:
        """
        Execute an MCP tool asynchronously with timeout.

        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool
            timeout: Timeout in seconds (default: 30.0)

        Returns:
            MCPToolResult with the operation result
        """
        try:
            # Use asyncio.wait_for for timeout
            return await asyncio.wait_for(
                self._execute_tool_internal(tool_name, arguments), timeout=timeout
            )
        except TimeoutError:
            LOGGER.error(f"MCP tool execution timed out after {timeout}s")
            return MCPToolResult(
                success=False, error=f"MCP request timed out after {timeout}s"
            )
        except Exception as e:
            LOGGER.error(f"MCP tool execution error: {e}")
            return MCPToolResult(success=False, error=str(e))

    async def _execute_tool_internal(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> MCPToolResult:
        """
        Internal method to execute MCP tool without timeout.

        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool

        Returns:
            MCPToolResult with the operation result
        """
        async with streamablehttp_client(self.server_url) as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                result = await session.call_tool(tool_name, arguments=arguments)

                # Parse the result content
                if result.content:
                    content = result.content[0]
                    if hasattr(content, "text"):
                        try:
                            data = json.loads(content.text)
                            if data.get("status") == "success":
                                return MCPToolResult(
                                    success=True,
                                    data=data.get("data", data),
                                    message=data.get("message", ""),
                                )
                            else:
                                return MCPToolResult(
                                    success=False,
                                    error=data.get("message", "Unknown error"),
                                )
                        except json.JSONDecodeError:
                            return MCPToolResult(
                                success=True, data=content.text, message=""
                            )

                return MCPToolResult(success=False, error="No content in response")

    async def _read_resource_async(self, resource_uri: str) -> str | None:
        """
        Read a resource from the MCP server asynchronously.

        Args:
            resource_uri: URI of the resource to read

        Returns:
            Resource content as string, or None on error
        """
        try:
            async with streamablehttp_client(self.server_url) as (
                read_stream,
                write_stream,
                _,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()

                    result = await session.read_resource(resource_uri)

                    if result.contents:
                        content = result.contents[0]
                        if hasattr(content, "text"):
                            return content.text

                    return None

        except Exception as e:
            LOGGER.error(f"MCP resource read error: {e}")
            return None

    def _run_async(self, coro):
        """
        Run an async coroutine synchronously.

        Handles the case where we might already be in an event loop.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            # We're in an async context, create a new loop in a thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        else:
            # No running loop, just run normally
            return asyncio.run(coro)

    # -------------------------------------------------------------------------
    # Synchronous public API
    # -------------------------------------------------------------------------

    def execute_sql_query(self, sql_query: str) -> MCPToolResult:
        """
        Execute a SQL query against the tracer database.

        Args:
            sql_query: SQL query string to execute

        Returns:
            MCPToolResult with query results or error
        """
        LOGGER.info(f"Executing SQL via MCP: {sql_query[:100]}...")
        return self._run_async(
            self._execute_tool_async("execute_sql_query", {"sql_query": sql_query})
        )

    def start_tracing(self, domain: str, directory: str = None) -> MCPToolResult:
        """
        Start tracing for a specific domain.

        Args:
            domain: Domain to trace (e.g., "file_system", "network")
            directory: Optional directory to trace

        Returns:
            MCPToolResult with operation status
        """
        args = {"domain": domain}
        if directory:
            args["directory"] = directory
        return self._run_async(self._execute_tool_async("start_tracing", args))

    def stop_tracing(self, domain: str, directory: str = None) -> MCPToolResult:
        """
        Stop tracing for a specific domain.

        Args:
            domain: Domain to stop tracing
            directory: Optional directory that was being traced

        Returns:
            MCPToolResult with operation status
        """
        args = {"domain": domain}
        if directory:
            args["directory"] = directory
        return self._run_async(self._execute_tool_async("stop_tracing", args))

    def list_domains(self) -> MCPToolResult:
        """
        List all available tracing domains.

        Returns:
            MCPToolResult with list of domains
        """
        return self._run_async(self._execute_tool_async("list_domains", {}))

    def list_tracers(self) -> MCPToolResult:
        """
        List all active tracers.

        Returns:
            MCPToolResult with list of active tracers
        """
        return self._run_async(self._execute_tool_async("list_tracers", {}))

    def initialize_database(self) -> MCPToolResult:
        """
        Initialize the tracer database.

        Returns:
            MCPToolResult with operation status
        """
        return self._run_async(self._execute_tool_async("initialize_database", {}))

    def get_schema(self, domain: str = "file_system") -> MCPSchema | None:
        """
        Get the database schema for a domain.

        Args:
            domain: Domain to get schema for (default: "file_system")

        Returns:
            MCPSchema with schema information, or None on error
        """
        # Check cache first
        if domain in self._schema_cache:
            return self._schema_cache[domain]

        # Try to read from MCP resource
        try:
            resource_uri = f"schema://{domain}"
            schema_text = self._run_async(self._read_resource_async(resource_uri))
        except Exception as e:
            LOGGER.warning(f"Could not read schema resource: {e}")
            schema_text = None

        # Use default schema if resource read fails
        if not schema_text:
            LOGGER.info(f"Using default schema for {domain}")
            schema_text = self._get_default_schema()

        schema = MCPSchema(domain=domain, schema_text=schema_text, tables=[domain])
        self._schema_cache[domain] = schema
        return schema

    def get_all_schemas(self) -> str:
        """
        Get all available database schemas as a formatted string.

        Returns:
            Formatted string containing all schemas
        """
        schemas = []

        # Try to get filesystem schema
        try:
            fs_schema = self.get_schema("file_system")
            if fs_schema:
                schemas.append(
                    f"=== {fs_schema.domain.upper()} ===\n{fs_schema.schema_text}"
                )
        except Exception as e:
            LOGGER.warning(f"Could not get file_system schema: {e}")

        # Try to get network schema if available
        try:
            net_schema = self.get_schema("network")
            if net_schema:
                schemas.append(
                    f"=== {net_schema.domain.upper()} ===\n{net_schema.schema_text}"
                )
        except Exception:
            # Network schema might not exist, that's ok
            pass

        if schemas:
            return "\n\n".join(schemas)
        else:
            return self._get_default_schema()

    def _get_default_schema(self) -> str:
        """Get a default schema description if MCP is unavailable."""
        return """Table: file_system
Columns:
- id: INTEGER (Primary Key)
- event: VARCHAR (values: 'created', 'modified', 'deleted', 'moved')
- name: VARCHAR (file or directory name)
- is_directory: BOOLEAN
- full_path: VARCHAR (complete file path)
- timestamp: DATETIME"""

    def test_connection(self) -> bool:
        """
        Test the connection to the MCP server.

        Returns:
            True if connection is successful, False otherwise
        """
        try:
            result = self.list_domains()
            return result.success
        except Exception as e:
            LOGGER.error(f"MCP connection test failed: {e}")
            return False


# Singleton instance for reuse
_mcp_client: MCPClientWrapper | None = None
_mcp_client_lock = threading.Lock()


def get_mcp_client() -> MCPClientWrapper:
    """
    Get or create a singleton MCPClientWrapper instance (thread-safe).

    Returns:
        The MCPClientWrapper instance
    """
    global _mcp_client
    if _mcp_client is None:
        with _mcp_client_lock:
            if _mcp_client is None:
                _mcp_client = MCPClientWrapper()
    return _mcp_client
