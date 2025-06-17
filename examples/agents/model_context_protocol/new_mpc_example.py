import asyncio

from ToolAgents import ToolRegistry
from ToolAgents.model_context_protocol.new_mcp_support import create_mcp_server_from_registry, MCPResourceContent, \
    MCPResource, run_mcp_server_stdio


def example_usage():
    """Example of how to use MCP integration with ToolAgent."""
    import sys
    from example_tools import (
        calculator_function_tool,
        current_datetime_function_tool,
        get_weather_function_tool,
        read_file_tool,
        write_file_tool,
        list_files_tool
    )

    # Create a tool registry and add tools
    registry = ToolRegistry()
    registry.add_tools([
        calculator_function_tool,
        current_datetime_function_tool,
        get_weather_function_tool,
        read_file_tool,
        write_file_tool,
        list_files_tool
    ])

    # Create MCP server from registry
    server = create_mcp_server_from_registry(registry, "My ToolAgent MCP Server")

    # Add a simple resource
    def get_system_info(uri: str) -> MCPResourceContent:
        return MCPResourceContent(
            uri=uri,
            mimeType="text/plain",
            text="System information: Python ToolAgent MCP Server"
        )

    server.add_resource(
        MCPResource(
            uri="system://info",
            name="System Information",
            description="Basic system information",
            mimeType="text/plain",
        ),
        get_system_info
    )
    if sys.platform == 'win32':
        # Set the policy to prevent "Event loop is closed" error on Windows - https://github.com/encode/httpx/issues/914
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # Run the server
    asyncio.run(run_mcp_server_stdio(server))

# Uncomment to run example
example_usage()