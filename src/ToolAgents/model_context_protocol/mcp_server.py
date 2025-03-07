from typing import List, Dict, Any, Optional, AsyncIterator
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP, Context
from mcp.types import Tool as MCPTool

from ToolAgents import FunctionTool, ToolRegistry


#class MCPFunctionToolServer

class MCPToolAgentServer:
    """
    An MCP server that exposes ToolAgent FunctionTools as MCP tools.

    This class bridges between the ToolAgent framework and the Model Context Protocol,
    allowing FunctionTools to be used with MCP-compatible clients like Claude.
    """

    def __init__(
            self,
            name: str,
            tool_registry: Optional[ToolRegistry] = None,
            function_tools: Optional[List[FunctionTool]] = None,
            dependencies: Optional[List[str]] = None,
    ):
        """
        Initialize the MCP server with ToolAgent tools.

        Args:
            name: The name of the MCP server
            tool_registry: Optional ToolRegistry containing FunctionTools
            function_tools: Optional list of individual FunctionTools to register
            dependencies: Optional list of package dependencies for the server
        """
        self.name = name
        self.tool_registry = tool_registry or ToolRegistry()
        if dependencies:
            self.mcp_server = FastMCP(name, dependencies=dependencies)
        else:
            self.mcp_server = FastMCP(name)

        # Register any provided function tools
        if function_tools:
            for tool in function_tools:
                self.tool_registry.add_tool(tool)

        # Register all tools with the MCP server
        self._register_tools()

    def _register_tools(self) -> None:
        """Register all FunctionTools from the tool registry with the MCP server."""
        for tool_name, function_tool in self.tool_registry.tools.items():
            self._register_tool(function_tool)

    def _register_tool(self, function_tool: FunctionTool) -> None:
        """
        Register a single FunctionTool with the MCP server.

        Args:
            function_tool: The FunctionTool to register
        """
        # Get the pydantic model from the function tool
        model = function_tool.model

        # Extract parameter information for the MCP tool
        param_info = model.model_json_schema()

        # Create a wrapper function that will execute the tool
        @self.mcp_server.tool()
        def tool_wrapper(**params):
            """Execute the function tool with the provided parameters."""
            result = function_tool.execute(params)
            return result

        # Rename the wrapper function to match the original tool name
        tool_wrapper.__name__ = model.__name__

        # Copy the docstring from the original model
        tool_wrapper.__doc__ = model.__doc__

    def add_tool(self, function_tool: FunctionTool) -> None:
        """
        Add a new FunctionTool to the server.

        Args:
            function_tool: The FunctionTool to add
        """
        self.tool_registry.add_tool(function_tool)
        self._register_tool(function_tool)

    def add_tools(self, function_tools: List[FunctionTool]) -> None:
        """
        Add multiple FunctionTools to the server.

        Args:
            function_tools: The list of FunctionTools to add
        """
        for tool in function_tools:
            self.add_tool(tool)

    def add_resource(self, uri_pattern: str):
        """
        Add a resource to the MCP server.

        Args:
            uri_pattern: The URI pattern for the resource

        Returns:
            A decorator that can be used to register a resource function
        """
        return self.mcp_server.resource(uri_pattern)

    def add_prompt(self):
        """
        Add a prompt to the MCP server.

        Returns:
            A decorator that can be used to register a prompt function
        """
        return self.mcp_server.prompt()

    def run(self):
        """Run the MCP server."""
        self.mcp_server.run()

    def install(self, name: Optional[str] = None):
        """
        Install the server in Claude Desktop.

        Args:
            name: Optional custom name for the server installation
        """
        # This would typically call the MCP CLI to install
        # Since we can't directly call it from here, we'll print instructions
        server_name = name or self.name
        print(f"To install this server in Claude Desktop, run:")
        print(f"mcp install your_server_file.py --name \"{server_name}\"")


class AsyncMCPToolAgentServer(MCPToolAgentServer):
    """
    An asynchronous MCP server that exposes ToolAgent AsyncFunctionTools as MCP tools.
    """

    def __init__(
            self,
            name: str,
            tool_registry: Optional[ToolRegistry] = None,
            function_tools: Optional[List[FunctionTool]] = None,
            dependencies: Optional[List[str]] = None,
            lifespan=None,
    ):
        """
        Initialize the async MCP server with ToolAgent tools.

        Args:
            name: The name of the MCP server
            tool_registry: Optional ToolRegistry containing FunctionTools
            function_tools: Optional list of individual FunctionTools to register
            dependencies: Optional list of package dependencies for the server
            lifespan: Optional lifespan context manager for the server
        """
        super().__init__(name, tool_registry, function_tools, dependencies)

        # Replace the FastMCP instance with one that has the lifespan
        if lifespan:
            self.mcp_server = FastMCP(name, dependencies=dependencies, lifespan=lifespan)
            # Re-register all tools after replacing the server
            self._register_tools()

    def _register_tool(self, function_tool: FunctionTool) -> None:
        """
        Register a single FunctionTool with the MCP server.

        Args:
            function_tool: The FunctionTool to register
        """
        # Get the pydantic model from the function tool
        model = function_tool.model

        # Extract parameter information for the MCP tool
        param_info = model.model_json_schema()

        # Create an async wrapper function that will execute the tool
        @self.mcp_server.tool()
        async def tool_wrapper(**params):
            """Execute the function tool with the provided parameters."""
            result = await function_tool.execute_async(params)
            return result

        # Rename the wrapper function to match the original tool name
        tool_wrapper.__name__ = model.__name__

        # Copy the docstring from the original model
        tool_wrapper.__doc__ = model.__doc__


# Example usage:
if __name__ == "__main__":
    # Create a sample function tool
    from pydantic import BaseModel


    class Calculator(BaseModel):
        """A simple calculator tool."""
        a: int
        b: int
        operation: str

        def run(self):
            """Perform the specified calculation."""
            if self.operation == "add":
                return self.a + self.b
            elif self.operation == "subtract":
                return self.a - self.b
            elif self.operation == "multiply":
                return self.a * self.b
            elif self.operation == "divide":
                return self.a / self.b
            else:
                return f"Unknown operation: {self.operation}"


    # Create a FunctionTool from the Calculator model
    calculator_tool = FunctionTool(Calculator)

    # Create an MCP server with the calculator tool
    server = MCPToolAgentServer("Calculator Server", function_tools=[calculator_tool])

    # Run the server
    server.run()
