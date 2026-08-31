"""Load MCP server tools into ToolAgents."""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from time import sleep
from typing import Any, Callable, Optional, Tuple

from ToolAgents import FunctionTool
from ToolAgents.tool_adapters.schemas import json_schema_to_pydantic_model


def _require_mcp():
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        from mcp.client.streamable_http import streamablehttp_client
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "MCP support requires the optional dependency: "
            "pip install ToolAgents[mcp]"
        ) from exc
    return ClientSession, StdioServerParameters, stdio_client, streamablehttp_client


class SessionManager:
    """Manage an MCP ClientSession for stdio or streamable HTTP servers."""

    def __init__(self, server_params, is_stdio_session=True, sampling_callback=None):
        self.server_params = server_params
        self.sampling_callback = sampling_callback
        self.read = None
        self.write = None
        self.session = None
        self.is_stdio_session = is_stdio_session
        self.exit_stack = AsyncExitStack()

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()

    async def connect(self):
        """Establish connection and initialize the MCP session."""

        if self.session is not None:
            return self

        ClientSession, _, stdio_client, streamablehttp_client = _require_mcp()
        if self.is_stdio_session:
            self.read, self.write = await self.exit_stack.enter_async_context(
                stdio_client(self.server_params)
            )
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(self.read, self.write)
            )
        else:
            self.read, self.write, _ = await self.exit_stack.enter_async_context(
                streamablehttp_client(**self.server_params)
            )
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(
                    self.read,
                    self.write,
                    sampling_callback=self.sampling_callback,
                )
            )

        await self.session.initialize()
        return self

    async def disconnect(self):
        """Close the session and transport contexts."""

        if self.session is None:
            return
        await self.exit_stack.aclose()
        self.session = None
        self.read = None
        self.write = None
        self.exit_stack = AsyncExitStack()

    async def list_prompts(self):
        self._ensure_connected()
        return await self.session.list_prompts()

    async def get_prompt(self, prompt_name: str, arguments: Optional[dict[str, Any]] = None):
        self._ensure_connected()
        return await self.session.get_prompt(prompt_name, arguments=arguments or {})

    async def list_resources(self):
        self._ensure_connected()
        return await self.session.list_resources()

    async def read_resource(self, path: str) -> Tuple[bytes, str]:
        self._ensure_connected()
        return await self.session.read_resource(path)

    async def list_tools(self):
        self._ensure_connected()
        return await self.session.list_tools()

    async def call_tool(self, tool_name: str, arguments: Optional[dict[str, Any]] = None):
        self._ensure_connected()
        return await self.session.call_tool(tool_name, arguments=arguments or {})

    def _ensure_connected(self):
        if self.session is None:
            raise RuntimeError(
                "Session is not connected. Call connect() first or use as context manager."
            )


class MCPTool:
    """Small wrapper around MCP tool metadata."""

    def __init__(self, name: str, description: str, input_schema: dict[str, Any]):
        self.name = name
        self.description = description
        self.inputSchema = input_schema

    def get_pydantic_input_model(self):
        return json_schema_to_pydantic_model(self.inputSchema)

    def __repr__(self):
        return (
            f"MCPTool(name={self.name!r}, description={self.description!r}, "
            f"inputSchema={self.inputSchema!r})"
        )


def _mcp_tool_schema(tool: Any) -> dict[str, Any]:
    return (
        getattr(tool, "inputSchema", None)
        or getattr(tool, "input_schema", None)
        or getattr(tool, "parameters", None)
        or {}
    )


def _sync_call_mcp_tool(
    server_params,
    is_stdio_session: bool,
    sampling_callback: Optional[Callable],
    tool_name: str,
    arguments: dict[str, Any],
):
    async def execute_tool():
        async with SessionManager(
            server_params=server_params,
            is_stdio_session=is_stdio_session,
            sampling_callback=sampling_callback,
        ) as session_mgr:
            return await session_mgr.call_tool(tool_name, arguments=arguments)

    loop = asyncio.get_event_loop_policy().new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(execute_tool())
        while loop.is_running():
            sleep(0.1)
        return result
    finally:
        loop.close()


async def load_mcp_tools_from_stdio(server_params, sampling_callback: Optional[Callable] = None):
    """Load tools from an MCP stdio server as ToolAgents FunctionTools."""

    async with SessionManager(
        server_params=server_params,
        is_stdio_session=True,
        sampling_callback=sampling_callback,
    ) as session_mgr:
        tools_result = await session_mgr.list_tools()

    function_tools = []
    for tool in tools_result.tools:
        mcp_tool = MCPTool(tool.name, tool.description or "", _mcp_tool_schema(tool))

        def tool_executor(_tool_name=tool.name, **kwargs):
            return _sync_call_mcp_tool(
                server_params,
                True,
                sampling_callback,
                _tool_name,
                kwargs,
            )

        function_tool = FunctionTool.from_pydantic_model_and_callable(
            mcp_tool.get_pydantic_input_model(),
            tool_executor,
        )
        function_tool.set_name(tool.name)
        function_tools.append(function_tool)
    return function_tools


async def load_mcp_tools_from_http(
    server_kwargs: dict[str, Any],
    sampling_callback: Optional[Callable] = None,
):
    """Load tools from an MCP streamable HTTP server as ToolAgents FunctionTools."""

    async with SessionManager(
        server_params=server_kwargs,
        is_stdio_session=False,
        sampling_callback=sampling_callback,
    ) as session_mgr:
        tools_result = await session_mgr.list_tools()

    function_tools = []
    for tool in tools_result.tools:
        mcp_tool = MCPTool(tool.name, tool.description or "", _mcp_tool_schema(tool))

        def tool_executor(_tool_name=tool.name, **kwargs):
            return _sync_call_mcp_tool(
                server_kwargs,
                False,
                sampling_callback,
                _tool_name,
                kwargs,
            )

        function_tool = FunctionTool.from_pydantic_model_and_callable(
            mcp_tool.get_pydantic_input_model(),
            tool_executor,
        )
        function_tool.set_name(tool.name)
        function_tools.append(function_tool)
    return function_tools


class MCPServerTools:
    """Compatibility loader for MCP tools."""

    def __init__(self):
        self.tools = None
        self.server_params = None
        self.session = None
        self.is_stdio_session = False
        self.sampling_callback = None

    async def load_from_stdio_server(self, server_params, sampling_callback: Optional[Callable] = None):
        self.sampling_callback = sampling_callback
        self.server_params = server_params
        self.is_stdio_session = True
        self.tools = await load_mcp_tools_from_stdio(server_params, sampling_callback)
        return self.tools

    async def load_from_http_server(
        self,
        server_kwargs: dict[str, Any],
        sampling_callback: Optional[Callable] = None,
    ):
        self.sampling_callback = sampling_callback
        self.server_params = server_kwargs
        self.is_stdio_session = False
        self.tools = await load_mcp_tools_from_http(server_kwargs, sampling_callback)
        return self.tools
