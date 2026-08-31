"""Serve NavigableMemory tools over MCP."""

from __future__ import annotations

from ToolAgents.tool_adapters.mcp_server import serve_tools_as_mcp

from navigable_memory_tools import create_tools


if __name__ == "__main__":
    serve_tools_as_mcp(
        create_tools(),
        name="navigable-memory",
        transport="streamable-http",
        port=8042,
    )
