"""Adapters for using ToolAgents tools outside the core agent loop."""

from .execution import (
    ToolExecutionResult,
    async_execute_tool,
    async_execute_tool_by_name,
    execute_tool,
    execute_tool_by_name,
    find_tool,
    normalize_tools,
)
from .schemas import (
    ToolAdapterMetadata,
    function_tool_to_input_schema,
    function_tool_to_mcp_metadata,
    json_schema_to_pydantic_model,
)
from .mcp_client import (
    MCPServerTools,
    MCPTool,
    SessionManager,
    load_mcp_tools_from_http,
    load_mcp_tools_from_stdio,
)
from .mcp_server import create_mcp_server, serve_tools_as_mcp

__all__ = [
    "MCPServerTools",
    "MCPTool",
    "SessionManager",
    "ToolExecutionResult",
    "ToolAdapterMetadata",
    "async_execute_tool",
    "async_execute_tool_by_name",
    "create_mcp_server",
    "execute_tool",
    "execute_tool_by_name",
    "find_tool",
    "function_tool_to_input_schema",
    "function_tool_to_mcp_metadata",
    "json_schema_to_pydantic_model",
    "load_mcp_tools_from_http",
    "load_mcp_tools_from_stdio",
    "normalize_tools",
    "serve_tools_as_mcp",
]
