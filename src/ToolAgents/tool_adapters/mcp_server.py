"""Expose ToolAgents tools as an MCP server."""

from __future__ import annotations

import inspect
from typing import Any, Literal

from pydantic_core import PydanticUndefined

from ToolAgents.function_tool import FunctionTool
from ToolAgents.tool_adapters.execution import (
    ToolCollection,
    async_execute_tool_by_name,
    format_tool_result,
    normalize_tools,
)
from ToolAgents.tool_adapters.schemas import function_tool_to_mcp_metadata


def _require_mcp_server_class():
    try:
        from mcp.server import FastMCP  # type: ignore

        return FastMCP
    except (ImportError, AttributeError):
        pass

    try:
        from mcp.server.fastmcp import FastMCP  # type: ignore

        return FastMCP
    except ImportError:
        pass

    try:
        from mcp.server.mcpserver.server import MCPServer

        return MCPServer
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "MCP server support requires the optional dependency: "
            "pip install ToolAgents[mcp]"
        ) from exc


def _signature_for_tool(tool: FunctionTool) -> inspect.Signature:
    parameters = []
    for field_name, field in tool.model.model_fields.items():
        default = inspect.Parameter.empty
        if not field.is_required():
            default = None if field.default is PydanticUndefined else field.default
        parameters.append(
            inspect.Parameter(
                field_name,
                inspect.Parameter.KEYWORD_ONLY,
                default=default,
                annotation=field.annotation or inspect.Parameter.empty,
            )
        )
    return inspect.Signature(parameters=parameters)


def create_mcp_tool_callable(
    tool: FunctionTool,
    tools: list[FunctionTool],
):
    """Create an async callable that MCP can inspect and execute."""

    metadata = function_tool_to_mcp_metadata(tool)

    async def call_tool(**kwargs: Any) -> str:
        result = await async_execute_tool_by_name(tools, metadata.name, kwargs)
        return format_tool_result(result)

    call_tool.__name__ = metadata.name
    call_tool.__doc__ = metadata.description
    call_tool.__signature__ = _signature_for_tool(tool)  # type: ignore[attr-defined]
    return call_tool


def create_mcp_server(
    tools: ToolCollection,
    name: str = "toolagents",
    **server_kwargs: Any,
):
    """Create an MCP server exposing ToolAgents tools."""

    server_class = _require_mcp_server_class()
    function_tools = normalize_tools(tools)
    server = server_class(name, **server_kwargs)

    for tool in function_tools:
        metadata = function_tool_to_mcp_metadata(tool)
        callable_tool = create_mcp_tool_callable(tool, function_tools)
        if hasattr(server, "add_tool"):
            server.add_tool(
                callable_tool,
                name=metadata.name,
                description=metadata.description,
            )
        else:
            server.tool(metadata.name, description=metadata.description)(callable_tool)
    return server


def serve_tools_as_mcp(
    tools: ToolCollection,
    name: str = "toolagents",
    transport: Literal["stdio", "sse", "streamable-http"] = "streamable-http",
    host: str = "127.0.0.1",
    port: int = 8042,
    **server_kwargs: Any,
) -> None:
    """Run an MCP server exposing ToolAgents tools."""

    server = create_mcp_server(tools, name=name)
    run_kwargs: dict[str, Any] = dict(server_kwargs)
    if transport in {"sse", "streamable-http"}:
        run_kwargs.setdefault("host", host)
        run_kwargs.setdefault("port", port)
    server.run(transport, **run_kwargs)
