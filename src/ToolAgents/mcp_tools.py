import asyncio
import json
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union

# Assuming these are imported in your original code
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client


class SessionManager:
    """
    A wrapper class for ClientSession that manages connection and provides
    access to tools, prompts, and resources.
    """

    def __init__(self, server_params, sampling_callback=None):
        self.server_params = server_params
        self.sampling_callback = sampling_callback
        self.read = None
        self.write = None
        self.session = None
        self._client_context = None
        self._session_context = None

    async def __aenter__(self):
        """Support using as an async context manager."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when exiting the context."""
        await self.disconnect()

    async def connect(self):
        """Establish connection and initialize session."""
        if self.session is not None:
            return self

        # Create and store the client context
        self._client_context = stdio_client(self.server_params)
        self.read, self.write = await self._client_context.__aenter__()

        # Create and store the session context
        self._session_context = ClientSession(
            self.read, self.write, sampling_callback=self.sampling_callback
        )
        self.session = await self._session_context.__aenter__()

        # Initialize the connection
        await self.session.initialize()
        return self

    async def disconnect(self):
        """Close the session and connection."""
        if self.session is None:
            return

        # Close session
        await self._session_context.__aexit__(None, None, None)

        # Close client
        await self._client_context.__aexit__(None, None, None)

        self.session = None
        self.read = None
        self.write = None
        self._client_context = None
        self._session_context = None

    async def list_prompts(self):
        """List available prompts."""
        self._ensure_connected()
        return await self.session.list_prompts()

    async def get_prompt(self, prompt_name: str, arguments: Optional[Dict[str, Any]] = None):
        """Get a specific prompt with optional arguments."""
        self._ensure_connected()
        return await self.session.get_prompt(prompt_name, arguments=arguments or {})

    async def list_resources(self):
        """List available resources."""
        self._ensure_connected()
        return await self.session.list_resources()

    async def read_resource(self, path: str) -> Tuple[bytes, str]:
        """Read a resource at the specified path."""
        self._ensure_connected()
        return await self.session.read_resource(path)

    async def list_tools(self):
        """List available tools."""
        self._ensure_connected()
        return await self.session.list_tools()

    async def call_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None):
        """Call a specific tool with optional arguments."""
        self._ensure_connected()
        return await self.session.call_tool(tool_name, arguments=arguments or {})

    def _ensure_connected(self):
        """Ensure the session is connected before operations."""
        if self.session is None:
            raise RuntimeError("Session is not connected. Call connect() first or use as context manager.")


# Example usage:
async def example_usage():
    # Create server parameters for stdio connection
    server_params = StdioServerParameters(
        command="uv",  # Executable
        args=[
        "run",
        "--with",
        "mcp[cli]",
        "mcp",
        "run",
        "H:\\MaxDev42\\ToolAgentsDev\\examples\\personal_mcp_tools.py"
      ],  # Optional command line arguments
        env=None,  # Optional environment variables
    )
    # Option 1: Using as a context manager (recommended)
    async with SessionManager(server_params) as session_mgr:
        prompts = await session_mgr.list_prompts()
        tools = await session_mgr.list_tools()
        print(tools)
        print(json.dumps(tools.tools[0].inputSchema, indent=2))

if __name__ == "__main__":
    import asyncio

    asyncio.run(example_usage())