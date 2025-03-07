import asyncio
from typing import Dict, List, Optional, Any, Callable
import re

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import mcp.types as types

from ToolAgents.function_tool import FunctionTool, ToolRegistry
from ToolAgents.utilities.pydantic_utilites import (
    create_dynamic_models_from_dictionaries,
    add_run_method_to_dynamic_model
)


class MCPAdapter:
    """
    Simple adapter for using MCP server tools with the ToolAgents framework.

    This class connects to an MCP server and converts its tools into FunctionTool
    instances that can be used by the ToolAgents framework.
    """

    def __init__(self, server_params: StdioServerParameters):
        """
        Initialize the adapter with server connection parameters.

        Args:
            server_params: The parameters needed to connect to the MCP server
        """
        self.server_params = server_params
        self.mcp_client = None
        self.initialized = False

    def initialize(self):
        """
        Initialize the connection to the MCP server synchronously.
        """
        if not self.initialized:
            asyncio.run(self._initialize_async())

    async def _initialize_async(self):
        """
        Initialize the connection to the MCP server asynchronously.
        """
        if not self.initialized:
            read_stream, write_stream = stdio_client(self.server_params)
            self.mcp_client = ClientSession(read_stream, write_stream)
            await self.mcp_client.initialize()
            self.initialized = True

    def get_tools(self) -> Dict[str, FunctionTool]:
        """
        Get all tools from the MCP server as FunctionTool instances.

        Returns:
            Dictionary of tool name to FunctionTool instance
        """
        return asyncio.run(self._get_tools_async())

    async def _get_tools_async(self) -> Dict[str, FunctionTool]:
        """
        Get all tools from the MCP server as FunctionTool instances asynchronously.

        Returns:
            Dictionary of tool name to FunctionTool instance
        """
        if not self.initialized:
            await self._initialize_async()

        tools = {}

        # Get MCP tools
        mcp_tools = await self.mcp_client.list_tools()
        for tool in mcp_tools:
            function_tool = await self._convert_tool(tool)
            tools[tool.name] = function_tool

        return tools

    def get_resources(self) -> Dict[str, FunctionTool]:
        """
        Get all resources from the MCP server as FunctionTool instances.

        Returns:
            Dictionary of resource name to FunctionTool instance
        """
        return asyncio.run(self._get_resources_async())

    async def _get_resources_async(self) -> Dict[str, FunctionTool]:
        """
        Get all resources from the MCP server as FunctionTool instances asynchronously.

        Returns:
            Dictionary of resource name to FunctionTool instance
        """
        if not self.initialized:
            await self._initialize_async()

        resources = {}

        # Get MCP resources
        mcp_resources = await self.mcp_client.list_resources()
        for resource in mcp_resources:
            function_tool = await self._convert_resource(resource)
            resource_name = self._get_resource_name(resource.uri_template)
            resources[resource_name] = function_tool

        return resources

    def get_prompts(self) -> Dict[str, FunctionTool]:
        """
        Get all prompts from the MCP server as FunctionTool instances.

        Returns:
            Dictionary of prompt name to FunctionTool instance
        """
        return asyncio.run(self._get_prompts_async())

    async def _get_prompts_async(self) -> Dict[str, FunctionTool]:
        """
        Get all prompts from the MCP server as FunctionTool instances asynchronously.

        Returns:
            Dictionary of prompt name to FunctionTool instance
        """
        if not self.initialized:
            await self._initialize_async()

        prompts = {}

        # Get MCP prompts
        mcp_prompts = await self.mcp_client.list_prompts()
        for prompt in mcp_prompts:
            function_tool = await self._convert_prompt(prompt)
            prompts[prompt.name] = function_tool

        return prompts

    def get_tool_registry(self) -> ToolRegistry:
        """
        Get a ToolRegistry containing all tools, resources, and prompts from the MCP server.

        Returns:
            ToolRegistry containing all MCP components
        """
        registry = ToolRegistry()

        # Add tools
        for tool in self.get_tools().values():
            registry.add_tool(tool)

        # Add resources
        for resource in self.get_resources().values():
            registry.add_tool(resource)

        # Add prompts
        for prompt in self.get_prompts().values():
            registry.add_tool(prompt)

        return registry

    def _get_resource_name(self, uri_template: str) -> str:
        """
        Generate a valid function name from a resource URI template.

        Args:
            uri_template: The URI template to convert

        Returns:
            A valid function name
        """
        return f"resource_{uri_template.replace('://', '_').replace('/', '_').replace('{', '').replace('}', '')}"

    async def _convert_tool(self, tool: types.Tool) -> FunctionTool:
        """
        Convert an MCP tool to a FunctionTool.

        Args:
            tool: MCP tool definition

        Returns:
            FunctionTool instance
        """
        # Create OpenAI-style tool definition
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or f"Tool: {tool.name}",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }

        # Add parameters
        for param in tool.arguments:
            # Determine type
            param_type = "string"  # Default
            if hasattr(param, 'type') and param.type:
                param_type = param.type

            openai_tool["function"]["parameters"]["properties"][param.name] = {
                "type": param_type,
                "description": param.description or f"Parameter: {param.name}"
            }

            # Add to required list if needed
            if param.required:
                openai_tool["function"]["parameters"]["required"].append(param.name)

        # Create a function that calls the MCP tool
        mcp_client = self.mcp_client  # Capture in closure

        async def tool_function(**kwargs):
            return await mcp_client.call_tool(tool.name, kwargs)

        # Create a normal function that wraps the async function
        def sync_tool_function(**kwargs):
            return asyncio.run(tool_function(**kwargs))

        # Create the FunctionTool
        models = create_dynamic_models_from_dictionaries([openai_tool])
        model = add_run_method_to_dynamic_model(models[0], sync_tool_function)
        return FunctionTool(model)

    async def _convert_resource(self, resource: types.Resource) -> FunctionTool:
        """
        Convert an MCP resource to a FunctionTool.

        Args:
            resource: MCP resource definition

        Returns:
            FunctionTool instance
        """
        # Extract parameters from URI template
        params = re.findall(r'\{([^}]+)\}', resource.uri_template)

        # Create OpenAI-style tool definition
        resource_name = self._get_resource_name(resource.uri_template)

        openai_tool = {
            "type": "function",
            "function": {
                "name": resource_name,
                "description": resource.description or f"Resource: {resource.uri_template}",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }

        # Add parameters
        for param in params:
            openai_tool["function"]["parameters"]["properties"][param] = {
                "type": "string",
                "description": f"Parameter: {param}"
            }
            openai_tool["function"]["parameters"]["required"].append(param)

        # Create a function that calls the MCP resource
        mcp_client = self.mcp_client  # Capture in closure
        uri_template = resource.uri_template  # Capture in closure

        async def resource_function(**kwargs):
            # Construct the URI by replacing template parameters
            uri = uri_template
            for param, value in kwargs.items():
                uri = uri.replace(f"{{{param}}}", str(value))

            # Read the resource
            content, mime_type = await mcp_client.read_resource(uri)
            return content

        # Create a normal function that wraps the async function
        def sync_resource_function(**kwargs):
            return asyncio.run(resource_function(**kwargs))

        # Create the FunctionTool
        models = create_dynamic_models_from_dictionaries([openai_tool])
        model = add_run_method_to_dynamic_model(models[0], sync_resource_function)
        return FunctionTool(model)

    async def _convert_prompt(self, prompt: types.Prompt) -> FunctionTool:
        """
        Convert an MCP prompt to a FunctionTool.

        Args:
            prompt: MCP prompt definition

        Returns:
            FunctionTool instance
        """
        # Create OpenAI-style tool definition
        openai_tool = {
            "type": "function",
            "function": {
                "name": prompt.name,
                "description": prompt.description or f"Prompt: {prompt.name}",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }

        # Add parameters
        for param in prompt.arguments:
            openai_tool["function"]["parameters"]["properties"][param.name] = {
                "type": "string",
                "description": param.description or f"Parameter: {param.name}"
            }

            # Add to required list if needed
            if param.required:
                openai_tool["function"]["parameters"]["required"].append(param.name)

        # Create a function that calls the MCP prompt
        mcp_client = self.mcp_client  # Capture in closure

        async def prompt_function(**kwargs):
            result = await mcp_client.get_prompt(prompt.name, kwargs)

            # Convert the prompt messages to a string
            prompt_text = ""
            for msg in result.messages:
                role = msg.role
                content = ""

                if hasattr(msg.content, 'text'):
                    content = msg.content.text
                elif isinstance(msg.content, types.TextContent):
                    content = msg.content.text

                prompt_text += f"{role}: {content}\n\n"

            return prompt_text

        # Create a normal function that wraps the async function
        def sync_prompt_function(**kwargs):
            return asyncio.run(prompt_function(**kwargs))

        # Create the FunctionTool
        models = create_dynamic_models_from_dictionaries([openai_tool])
        model = add_run_method_to_dynamic_model(models[0], sync_prompt_function)
        return FunctionTool(model)
