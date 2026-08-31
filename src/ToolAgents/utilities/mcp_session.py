"""Compatibility imports for MCP client helpers.

New code should import from ToolAgents.tool_adapters.mcp_client.
"""

from ToolAgents.tool_adapters.mcp_client import (
    MCPServerTools,
    MCPTool,
    SessionManager,
    load_mcp_tools_from_http,
    load_mcp_tools_from_stdio,
)

__all__ = [
    "SessionManager",
    "MCPTool",
    "MCPServerTools",
    "load_mcp_tools_from_stdio",
    "load_mcp_tools_from_http",
]
