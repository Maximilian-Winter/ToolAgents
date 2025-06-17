# complete_mcp_example.py
import asyncio
import sys
from typing import Dict, Any

from ToolAgents import ToolRegistry, FunctionTool
from ToolAgents.model_context_protocol.new_mcp_support import (
    create_mcp_server_from_registry,
    run_mcp_server_stdio,
    MCPClient,
    MCPStdioTransport
)



# Example tools (you'll need to import your actual tools)
def create_example_tools():
    """Create some example tools for demonstration."""
    from example_tools import current_datetime_function_tool
    from example_tools import  calculator_function_tool
    from example_tools import get_weather_function_tool


    return [
        current_datetime_function_tool,
        calculator_function_tool,
        get_weather_function_tool
    ]


async def run_mcp_server_example():
    """Example of running an MCP server."""

    # Create tools and registry
    tools = create_example_tools()
    registry = ToolRegistry()
    registry.add_tools(tools)

    # Create enhanced MCP server
    server = EnhancedMCPServer("Enhanced ToolAgent MCP Server", "1.0.0")
    server.add_tools_from_registry(registry)

    # Add middleware for logging
    def log_requests(message):
        print(f"Processing request: {message.method}")
        return message

    server.add_middleware(log_requests)

    # Add tool execution hook
    def tool_execution_hook(phase, tool_name, arguments, result=None):
        if phase == "pre":
            print(f"About to execute {tool_name} with {arguments}")
        else:
            print(f"Executed {tool_name}, result: {result}")

    for tool in tools:
        server.add_tool_hook(tool.model.__name__, tool_execution_hook)

    # Run server
    transport = MCPStdioTransport()
    async with mcp_server_context(server, transport):
        pass


async def run_mcp_client_example():
    """Example of using MCP client to connect to remote tools."""

    # This would connect to a remote MCP server
    client = MCPClient("ToolAgent MCP Client")

    # You would connect to a real server here
    # transport = MCPStdioTransport()  # or MCPWebSocketTransport("ws://localhost:8080")
    # await client.connect(transport)

    # Create registry that can pull from MCP servers
    mcp_registry = MCPToolRegistry()

    # Add MCP server tools to registry
    # await mcp_registry.add_mcp_server("remote_server", client, "remote_")

    # Get the registry with all tools (local + MCP)
    # combined_registry = mcp_registry.get_registry()

    print("MCP client example (would connect to real server)")


def main():
    """Main function to run examples."""
    if len(sys.argv) > 1 and sys.argv[1] == "client":
        asyncio.run(run_mcp_client_example())
    else:
        # Set event loop policy for Windows
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        # Run server
        asyncio.run(run_mcp_server_example())


if __name__ == "__main__":
    main()
