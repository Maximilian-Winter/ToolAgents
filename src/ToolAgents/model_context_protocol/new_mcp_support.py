"""
MCP (Model Context Protocol) integration for ToolAgent framework.

This module provides MCP server and client implementations that work with
the existing FunctionTool and ToolRegistry classes.
"""

import asyncio
import json
import logging
import platform
import queue
import sys
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, AsyncGenerator
from urllib.parse import urlparse
import uuid

from pydantic import BaseModel, Field

# Import your existing classes
from ToolAgents.function_tool import FunctionTool, ToolRegistry


# MCP Protocol Data Models
class MCPMessageType(Enum):
    """MCP message types as defined in the protocol."""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"


@dataclass
class MCPMessage:
    """Base MCP message structure."""
    jsonrpc: str = "2.0"
    id: Optional[Union[str, int]] = None
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None


class MCPError(Exception):
    """MCP-specific error."""

    def __init__(self, code: int, message: str, data: Optional[Any] = None):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f"MCP Error {code}: {message}")


class MCPErrorCodes:
    """Standard MCP error codes."""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603


# MCP Resource Models
class MCPResource(BaseModel):
    """MCP resource definition."""
    uri: str = Field(..., description="Unique resource identifier")
    name: str = Field(..., description="Human-readable resource name")
    description: Optional[str] = Field(None, description="Resource description")
    mimeType: Optional[str] = Field(None, description="MIME type of the resource")


class MCPResourceContent(BaseModel):
    """MCP resource content."""
    uri: str
    mimeType: str
    text: Optional[str] = None
    blob: Optional[bytes] = None


# MCP Tool Models
class MCPTool(BaseModel):
    """MCP tool definition."""
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    inputSchema: Dict[str, Any] = Field(..., description="JSON schema for tool input")


class MCPToolCall(BaseModel):
    """MCP tool call request."""
    name: str
    arguments: Dict[str, Any]


class MCPToolResult(BaseModel):
    """MCP tool call result."""
    content: List[Dict[str, Any]]
    isError: bool = False


# MCP Prompt Models
class MCPPrompt(BaseModel):
    """MCP prompt template."""
    name: str = Field(..., description="Prompt name")
    description: str = Field(..., description="Prompt description")
    arguments: Optional[List[Dict[str, Any]]] = Field(None, description="Prompt arguments schema")


# Transport Layer Abstraction
class MCPTransport(ABC):
    """Abstract base class for MCP transport implementations."""

    @abstractmethod
    async def send(self, message: MCPMessage) -> None:
        """Send a message through the transport."""
        pass

    @abstractmethod
    async def receive(self) -> AsyncGenerator[MCPMessage, None]:
        """Receive messages from the transport."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the transport connection."""
        pass


class MCPStdioTransport(MCPTransport):
    """Standard input/output transport for MCP with cross-platform support."""

    def __init__(self):
        self._closed = False
        self._initialized = False

        # Platform-specific attributes
        if platform.system() == "Windows":
            self.input_queue = None
            self.output_queue = None
            self.input_thread = None
            self.output_thread = None
            self.executor = None
        else:
            self.reader: Optional[asyncio.StreamReader] = None
            self.writer: Optional[asyncio.StreamWriter] = None

    async def initialize(self):
        """Initialize stdio transport with platform-specific handling."""
        if self._initialized:
            return

        if platform.system() == "Windows":
            await self._init_windows()
        else:
            await self._init_unix()

        self._initialized = True

    async def _init_windows(self):
        """Windows-specific initialization using threading."""
        import concurrent.futures

        self.input_queue = asyncio.Queue()
        self.output_queue = queue.Queue()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

        # Start background threads for I/O
        self.input_thread = threading.Thread(target=self._input_worker, daemon=True)
        self.output_thread = threading.Thread(target=self._output_worker, daemon=True)

        self.input_thread.start()
        self.output_thread.start()

    async def _init_unix(self):
        """Unix-specific initialization using standard asyncio."""
        self.reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(self.reader)
        loop = asyncio.get_event_loop()
        transport, _ = await loop.connect_read_pipe(lambda: protocol, sys.stdin)
        self.writer = asyncio.StreamWriter(sys.stdout, None, self.reader, loop)

    def _input_worker(self):
        """Worker thread to read from stdin (Windows only)."""
        try:
            loop = asyncio.new_event_loop()
            while not self._closed:
                try:
                    line = sys.stdin.readline()
                    if not line:
                        break

                    # Get the main event loop and schedule the coroutine
                    main_loop = asyncio.get_event_loop()
                    asyncio.run_coroutine_threadsafe(
                        self.input_queue.put(line.strip()),
                        main_loop
                    )
                except Exception as e:
                    # Handle errors gracefully
                    if not self._closed:
                        print(f"Input worker error: {e}", file=sys.stderr)
                    break
        except Exception as e:
            if not self._closed:
                print(f"Input worker initialization error: {e}", file=sys.stderr)

    def _output_worker(self):
        """Worker thread to write to stdout (Windows only)."""
        try:
            while not self._closed:
                try:
                    message = self.output_queue.get(timeout=0.1)
                    if message is None:  # Sentinel to stop
                        break
                    sys.stdout.write(message)
                    sys.stdout.flush()
                    self.output_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    if not self._closed:
                        print(f"Output worker error: {e}", file=sys.stderr)
                    break
        except Exception as e:
            if not self._closed:
                print(f"Output worker initialization error: {e}", file=sys.stderr)

    async def send(self, message: MCPMessage) -> None:
        """Send message to stdout."""
        if self._closed:
            raise MCPError(MCPErrorCodes.INTERNAL_ERROR, "Transport is closed")

        if not self._initialized:
            await self.initialize()

        message_dict = {
            "jsonrpc": message.jsonrpc,
        }

        if message.id is not None:
            message_dict["id"] = message.id
        if message.method is not None:
            message_dict["method"] = message.method
        if message.params is not None:
            message_dict["params"] = message.params
        if message.result is not None:
            message_dict["result"] = message.result
        if message.error is not None:
            message_dict["error"] = message.error

        json_str = json.dumps(message_dict)
        message_with_newline = f"{json_str}\n"

        if platform.system() == "Windows":
            # Use threading approach for Windows
            self.output_queue.put(message_with_newline)
        else:
            # Use standard asyncio for Unix
            if not self.writer:
                raise MCPError(MCPErrorCodes.INTERNAL_ERROR, "Writer not initialized")
            self.writer.write(message_with_newline.encode())
            await self.writer.drain()

    async def receive(self) -> AsyncGenerator[MCPMessage, None]:
        """Receive messages from stdin."""
        if not self._initialized:
            await self.initialize()

        while not self._closed:
            try:
                if platform.system() == "Windows":
                    # Use threading approach for Windows
                    line = await self.input_queue.get()
                    if not line:
                        break
                else:
                    # Use standard asyncio for Unix
                    if not self.reader:
                        raise MCPError(MCPErrorCodes.INTERNAL_ERROR, "Reader not initialized")
                    line_bytes = await self.reader.readline()
                    if not line_bytes:
                        break
                    line = line_bytes.decode().strip()

                if not line:
                    continue

                data = json.loads(line)
                message = MCPMessage(
                    jsonrpc=data.get("jsonrpc", "2.0"),
                    id=data.get("id"),
                    method=data.get("method"),
                    params=data.get("params"),
                    result=data.get("result"),
                    error=data.get("error")
                )
                yield message

            except json.JSONDecodeError as e:
                error_message = MCPMessage(
                    error={
                        "code": MCPErrorCodes.PARSE_ERROR,
                        "message": f"JSON parse error: {str(e)}"
                    }
                )
                yield error_message
            except Exception as e:
                if not self._closed:
                    error_message = MCPMessage(
                        error={
                            "code": MCPErrorCodes.INTERNAL_ERROR,
                            "message": f"Transport error: {str(e)}"
                        }
                    )
                    yield error_message

    async def close(self) -> None:
        """Close the transport."""
        self._closed = True

        if platform.system() == "Windows":
            # Clean up Windows threading resources
            if self.output_queue:
                self.output_queue.put(None)  # Sentinel to stop output worker

            # Wait for threads to finish (with timeout)
            if self.input_thread and self.input_thread.is_alive():
                self.input_thread.join(timeout=1.0)
            if self.output_thread and self.output_thread.is_alive():
                self.output_thread.join(timeout=1.0)

            if self.executor:
                self.executor.shutdown(wait=False)
        else:
            # Clean up Unix asyncio resources
            if self.writer:
                self.writer.close()
                await self.writer.wait_closed()


# MCP Server Implementation
class MCPServer:
    """MCP Server implementation that integrates with ToolAgent framework."""

    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.transport: Optional[MCPTransport] = None
        self.tool_registry = ToolRegistry()
        self.resources: Dict[str, MCPResource] = {}
        self.prompts: Dict[str, MCPPrompt] = {}
        self.resource_handlers: Dict[str, Callable[[str], MCPResourceContent]] = {}
        self.capabilities = {
            "tools": {},
            "resources": {},
            "prompts": {},
            "logging": {}
        }
        self.logger = logging.getLogger(f"mcp.server.{name}")

    def add_tool_from_function_tool(self, function_tool: FunctionTool) -> None:
        """Add a tool from existing FunctionTool."""
        self.tool_registry.add_tool(function_tool)
        self.capabilities["tools"] = {"listChanged": True}

    def add_tools_from_registry(self, registry: ToolRegistry) -> None:
        """Add all tools from an existing ToolRegistry."""
        for tool in registry.get_tools():
            self.tool_registry.add_tool(tool)
        self.capabilities["tools"] = {"listChanged": True}

    def add_resource(self, resource: MCPResource, handler: Callable[[str], MCPResourceContent]) -> None:
        """Add a resource with its content handler."""
        self.resources[resource.uri] = resource
        self.resource_handlers[resource.uri] = handler
        self.capabilities["resources"] = {"subscribe": True, "listChanged": True}

    def add_prompt(self, prompt: MCPPrompt) -> None:
        """Add a prompt template."""
        self.prompts[prompt.name] = prompt
        self.capabilities["prompts"] = {"listChanged": True}

    async def handle_message(self, message: MCPMessage) -> Optional[MCPMessage]:
        """Handle incoming MCP message."""
        try:
            if message.method == "initialize":
                return await self._handle_initialize(message)
            elif message.method == "tools/list":
                return await self._handle_tools_list(message)
            elif message.method == "tools/call":
                return await self._handle_tools_call(message)
            elif message.method == "resources/list":
                return await self._handle_resources_list(message)
            elif message.method == "resources/read":
                return await self._handle_resources_read(message)
            elif message.method == "prompts/list":
                return await self._handle_prompts_list(message)
            elif message.method == "prompts/get":
                return await self._handle_prompts_get(message)
            else:
                return MCPMessage(
                    id=message.id,
                    error={
                        "code": MCPErrorCodes.METHOD_NOT_FOUND,
                        "message": f"Method '{message.method}' not found"
                    }
                )
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
            return MCPMessage(
                id=message.id,
                error={
                    "code": MCPErrorCodes.INTERNAL_ERROR,
                    "message": str(e)
                }
            )

    async def _handle_initialize(self, message: MCPMessage) -> MCPMessage:
        """Handle initialization request."""
        return MCPMessage(
            id=message.id,
            result={
                "protocolVersion": "2024-11-05",
                "capabilities": self.capabilities,
                "serverInfo": {
                    "name": self.name,
                    "version": self.version
                }
            }
        )

    async def _handle_tools_list(self, message: MCPMessage) -> MCPMessage:
        """Handle tools list request."""
        tools = []
        for tool in self.tool_registry.get_tools():
            anthropic_tool = tool.to_anthropic_tool()
            mcp_tool = MCPTool(
                name=anthropic_tool["name"],
                description=anthropic_tool["description"],
                inputSchema=anthropic_tool["input_schema"]
            )
            tools.append(mcp_tool.model_dump())

        return MCPMessage(
            id=message.id,
            result={"tools": tools}
        )

    async def _handle_tools_call(self, message: MCPMessage) -> MCPMessage:
        """Handle tool call request."""
        if not message.params or "name" not in message.params:
            return MCPMessage(
                id=message.id,
                error={
                    "code": MCPErrorCodes.INVALID_PARAMS,
                    "message": "Missing tool name in parameters"
                }
            )

        tool_name = message.params["name"]
        arguments = message.params.get("arguments", {})

        try:
            tool = self.tool_registry.get_tool(tool_name)
            result = await tool.execute_async(arguments)

            # Format result for MCP
            content = [{"type": "text", "text": str(result)}]

            return MCPMessage(
                id=message.id,
                result={
                    "content": content,
                    "isError": False
                }
            )
        except KeyError:
            return MCPMessage(
                id=message.id,
                error={
                    "code": MCPErrorCodes.METHOD_NOT_FOUND,
                    "message": f"Tool '{tool_name}' not found"
                }
            )
        except Exception as e:
            return MCPMessage(
                id=message.id,
                result={
                    "content": [{"type": "text", "text": f"Error: {str(e)}"}],
                    "isError": True
                }
            )

    async def _handle_resources_list(self, message: MCPMessage) -> MCPMessage:
        """Handle resources list request."""
        resources = [resource.model_dump() for resource in self.resources.values()]
        return MCPMessage(
            id=message.id,
            result={"resources": resources}
        )

    async def _handle_resources_read(self, message: MCPMessage) -> MCPMessage:
        """Handle resource read request."""
        if not message.params or "uri" not in message.params:
            return MCPMessage(
                id=message.id,
                error={
                    "code": MCPErrorCodes.INVALID_PARAMS,
                    "message": "Missing resource URI in parameters"
                }
            )

        uri = message.params["uri"]

        if uri not in self.resource_handlers:
            return MCPMessage(
                id=message.id,
                error={
                    "code": MCPErrorCodes.METHOD_NOT_FOUND,
                    "message": f"Resource '{uri}' not found"
                }
            )

        try:
            content = self.resource_handlers[uri](uri)
            return MCPMessage(
                id=message.id,
                result={
                    "contents": [content.model_dump()]
                }
            )
        except Exception as e:
            return MCPMessage(
                id=message.id,
                error={
                    "code": MCPErrorCodes.INTERNAL_ERROR,
                    "message": f"Error reading resource: {str(e)}"
                }
            )

    async def _handle_prompts_list(self, message: MCPMessage) -> MCPMessage:
        """Handle prompts list request."""
        prompts = [prompt.model_dump() for prompt in self.prompts.values()]
        return MCPMessage(
            id=message.id,
            result={"prompts": prompts}
        )

    async def _handle_prompts_get(self, message: MCPMessage) -> MCPMessage:
        """Handle prompt get request."""
        if not message.params or "name" not in message.params:
            return MCPMessage(
                id=message.id,
                error={
                    "code": MCPErrorCodes.INVALID_PARAMS,
                    "message": "Missing prompt name in parameters"
                }
            )

        prompt_name = message.params["name"]

        if prompt_name not in self.prompts:
            return MCPMessage(
                id=message.id,
                error={
                    "code": MCPErrorCodes.METHOD_NOT_FOUND,
                    "message": f"Prompt '{prompt_name}' not found"
                }
            )

        # This is a simplified implementation - you'd want to handle prompt arguments
        prompt = self.prompts[prompt_name]
        return MCPMessage(
            id=message.id,
            result={
                "description": prompt.description,
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": f"Execute prompt: {prompt.name}"
                        }
                    }
                ]
            }
        )

    async def run(self, transport: MCPTransport) -> None:
        """Run the MCP server with the given transport."""
        self.transport = transport

        try:
            async for message in transport.receive():
                if message.error:
                    self.logger.error(f"Received error message: {message.error}")
                    continue

                response = await self.handle_message(message)
                if response:
                    await transport.send(response)
        except Exception as e:
            self.logger.error(f"Server error: {e}")
        finally:
            await transport.close()


# MCP Client Implementation
class MCPClient:
    """MCP Client implementation for connecting to MCP servers."""

    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.transport: Optional[MCPTransport] = None
        self.server_capabilities: Dict[str, Any] = {}
        self.request_id = 0
        self.pending_requests: Dict[Union[str, int], asyncio.Future] = {}
        self.logger = logging.getLogger(f"mcp.client.{name}")

    def _next_request_id(self) -> int:
        """Generate next request ID."""
        self.request_id += 1
        return self.request_id

    async def connect(self, transport: MCPTransport) -> None:
        """Connect to MCP server using transport."""
        self.transport = transport

        # Start message handling task
        asyncio.create_task(self._handle_messages())

        # Initialize connection
        await self._initialize()

    async def _initialize(self) -> None:
        """Initialize connection with server."""
        request_id = self._next_request_id()
        message = MCPMessage(
            id=request_id,
            method="initialize",
            params={
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "roots": {"listChanged": True},
                    "sampling": {}
                },
                "clientInfo": {
                    "name": self.name,
                    "version": self.version
                }
            }
        )

        response = await self._send_request(message)
        if response.error:
            raise MCPError(
                response.error["code"],
                response.error["message"],
                response.error.get("data")
            )

        self.server_capabilities = response.result.get("capabilities", {})
        self.logger.info(f"Connected to MCP server with capabilities: {self.server_capabilities}")

    async def _handle_messages(self) -> None:
        """Handle incoming messages from server."""
        if not self.transport:
            return

        async for message in self.transport.receive():
            if message.id and message.id in self.pending_requests:
                # This is a response to our request
                future = self.pending_requests.pop(message.id)
                future.set_result(message)
            else:
                # This might be a notification or unsolicited message
                self.logger.info(f"Received unsolicited message: {message}")

    async def _send_request(self, message: MCPMessage) -> MCPMessage:
        """Send request and wait for response."""
        if not self.transport:
            raise MCPError(MCPErrorCodes.INTERNAL_ERROR, "Not connected to server")

        future = asyncio.Future()
        self.pending_requests[message.id] = future

        await self.transport.send(message)

        try:
            response = await asyncio.wait_for(future, timeout=30.0)
            return response
        except asyncio.TimeoutError:
            self.pending_requests.pop(message.id, None)
            raise MCPError(MCPErrorCodes.INTERNAL_ERROR, "Request timeout")

    async def list_tools(self) -> List[MCPTool]:
        """List available tools from server."""
        request_id = self._next_request_id()
        message = MCPMessage(
            id=request_id,
            method="tools/list"
        )

        response = await self._send_request(message)
        if response.error:
            raise MCPError(
                response.error["code"],
                response.error["message"],
                response.error.get("data")
            )

        tools = []
        for tool_data in response.result.get("tools", []):
            tools.append(MCPTool(**tool_data))

        return tools

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> MCPToolResult:
        """Call a tool on the server."""
        request_id = self._next_request_id()
        message = MCPMessage(
            id=request_id,
            method="tools/call",
            params={
                "name": name,
                "arguments": arguments
            }
        )

        response = await self._send_request(message)
        if response.error:
            raise MCPError(
                response.error["code"],
                response.error["message"],
                response.error.get("data")
            )

        return MCPToolResult(**response.result)

    async def list_resources(self) -> List[MCPResource]:
        """List available resources from server."""
        request_id = self._next_request_id()
        message = MCPMessage(
            id=request_id,
            method="resources/list"
        )

        response = await self._send_request(message)
        if response.error:
            raise MCPError(
                response.error["code"],
                response.error["message"],
                response.error.get("data")
            )

        resources = []
        for resource_data in response.result.get("resources", []):
            resources.append(MCPResource(**resource_data))

        return resources

    async def read_resource(self, uri: str) -> List[MCPResourceContent]:
        """Read a resource from the server."""
        request_id = self._next_request_id()
        message = MCPMessage(
            id=request_id,
            method="resources/read",
            params={"uri": uri}
        )

        response = await self._send_request(message)
        if response.error:
            raise MCPError(
                response.error["code"],
                response.error["message"],
                response.error.get("data")
            )

        contents = []
        for content_data in response.result.get("contents", []):
            contents.append(MCPResourceContent(**content_data))

        return contents

    async def close(self) -> None:
        """Close the client connection."""
        if self.transport:
            await self.transport.close()


# Utility functions to integrate existing ToolAgent components
def create_mcp_server_from_registry(
        registry: ToolRegistry,
        server_name: str = "ToolAgent MCP Server"
) -> MCPServer:
    """Create an MCP server from an existing ToolRegistry."""
    server = MCPServer(server_name)
    server.add_tools_from_registry(registry)
    return server


async def run_mcp_server_stdio(server: MCPServer) -> None:
    """Run MCP server using stdio transport."""
    transport = MCPStdioTransport()
    await server.run(transport)

