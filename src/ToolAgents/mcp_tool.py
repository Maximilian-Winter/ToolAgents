import json
import requests
from typing import Any, Dict, List, Optional, Union, Callable

from pydantic import BaseModel, Field

from ToolAgents.function_tool import FunctionTool, BaseProcessor, PreProcessor, PostProcessor


class MCPToolDefinition(BaseModel):
    """
    Represents a Model Context Protocol tool definition.
    
    Attributes:
        name: The name of the tool
        description: Description of what the tool does
        input_schema: JSON schema describing the expected input format
        server_url: URL of the MCP server hosting this tool
    """
    name: str = Field(..., description="The name of the tool")
    description: str = Field(..., description="Description of what the tool does")
    input_schema: Dict[str, Any] = Field(..., description="JSON schema describing the expected input format")
    server_url: str = Field(..., description="URL of the MCP server hosting this tool")


class MCPTool:
    """
    Class representing a Model Context Protocol tool.
    
    Allows for integration with MCP-compatible servers and their tools.
    
    Args:
        tool_definition: Definition of the MCP tool
        pre_processors: Single preprocessor or list of preprocessors to apply before tool execution
        post_processors: Single postprocessor or list of postprocessors to apply after tool execution
        debug_mode: Enable debug mode to print parameters and responses
        timeout: Request timeout in seconds
    """
    
    def __init__(
        self,
        tool_definition: MCPToolDefinition,
        pre_processors: Union[PreProcessor, List[PreProcessor], Callable, List[Callable], None] = None,
        post_processors: Union[PostProcessor, List[PostProcessor], Callable, List[Callable], None] = None,
        debug_mode: bool = False,
        timeout: int = 30,
    ):
        self.tool_definition = tool_definition
        self.pre_processors = FunctionTool._normalize_processors(pre_processors)
        self.post_processors = FunctionTool._normalize_processors(post_processors)
        self.debug_mode = debug_mode
        self.timeout = timeout
        
        # Create a Pydantic model based on the input schema to enable validation
        self._create_input_model()
    
    def _create_input_model(self):
        """Create a Pydantic model from the tool's input schema"""
        from ToolAgents.function_tool import convert_dictionary_to_pydantic_model
        
        # Create a model for validation purposes
        schema_dict = {
            "type": "object",
            "properties": self.tool_definition.input_schema.get("properties", {}),
            "required": self.tool_definition.input_schema.get("required", [])
        }
        
        self.input_model = convert_dictionary_to_pydantic_model(
            schema_dict, 
            f"{self.tool_definition.name}Input"
        )
    
    def execute(self, parameters: Dict[str, Any]) -> Any:
        """
        Execute the MCP tool with the given parameters.
        
        Args:
            parameters: Input parameters for the tool
            
        Returns:
            The result from the tool execution
        """
        if self.debug_mode:
            print("Input parameters:")
            print(json.dumps(parameters, indent=4))
        
        # Apply pre-processors in sequence
        processed_params = parameters
        for processor in self.pre_processors:
            try:
                processed_params = processor(processed_params)
                if self.debug_mode:
                    print(f"After {processor.__class__.__name__}:")
                    print(json.dumps(processed_params, indent=4))
            except Exception as e:
                error_msg = f"Error in {processor.__class__.__name__}: {str(e)}"
                print(error_msg)
                return error_msg
        
        # Validate parameters against the input model
        try:
            # This will raise validation errors if parameters don't match the schema
            validated_params = self.input_model(**processed_params)
            
            # Now make the actual call to the MCP server
            result = self._call_mcp_server(processed_params)
            
        except Exception as e:
            error_msg = f"Error in tool execution: {str(e)}"
            print(error_msg)
            return error_msg
        
        # Apply post-processors in sequence
        processed_result = result
        for processor in self.post_processors:
            try:
                processed_result = processor(processed_result)
                if self.debug_mode:
                    print(f"After {processor.__class__.__name__}:")
                    print(processed_result)
            except Exception as e:
                error_msg = f"Error in {processor.__class__.__name__}: {str(e)}"
                print(error_msg)
                return error_msg
        
        return processed_result
    
    def _call_mcp_server(self, parameters: Dict[str, Any]) -> Any:
        """
        Make the actual API call to the MCP server.
        
        Args:
            parameters: Validated input parameters
            
        Returns:
            The response from the MCP server
        """
        # Construct the MCP API endpoint URL
        endpoint = f"{self.tool_definition.server_url}/tools/{self.tool_definition.name}"
        
        # Prepare the request payload
        payload = {
            "parameters": parameters
        }
        
        # Make the HTTP request
        try:
            response = requests.post(
                endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout
            )
            
            # Check if the request was successful
            response.raise_for_status()
            
            # Parse and return the response
            result = response.json()
            
            if self.debug_mode:
                print("MCP Server Response:")
                print(json.dumps(result, indent=4))
            
            # Extract the actual result from the response
            if "result" in result:
                return result["result"]
            else:
                return result
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Error communicating with MCP server: {str(e)}"
            print(error_msg)
            return error_msg
    
    def add_pre_processor(
        self,
        processor: Union[PreProcessor, Callable],
        position: int = None
    ) -> None:
        """
        Add a new preprocessor to the MCP tool.
        
        Args:
            processor: Preprocessor to add (either PreProcessor instance or callable)
            position: Optional position to insert the processor (None = append to end)
        """
        normalized = FunctionTool._normalize_processors([processor])[0]
        if position is None:
            self.pre_processors.append(normalized)
        else:
            self.pre_processors.insert(position, normalized)
    
    def add_post_processor(
        self,
        processor: Union[PostProcessor, Callable],
        position: int = None
    ) -> None:
        """
        Add a new postprocessor to the MCP tool.
        
        Args:
            processor: Postprocessor to add (either PostProcessor instance or callable)
            position: Optional position to insert the processor (None = append to end)
        """
        normalized = FunctionTool._normalize_processors([processor])[0]
        if position is None:
            self.post_processors.append(normalized)
        else:
            self.post_processors.insert(position, normalized)
    
    def to_openai_tool(self) -> Dict[str, Any]:
        """
        Convert the MCP tool to an OpenAI tool format.
        
        Returns:
            An OpenAI tool specification
        """
        return {
            "type": "function",
            "function": {
                "name": self.tool_definition.name,
                "description": self.tool_definition.description,
                "parameters": self.tool_definition.input_schema
            }
        }
    
    def to_anthropic_tool(self) -> Dict[str, Any]:
        """
        Convert the MCP tool to an Anthropic tool format.
        
        Returns:
            An Anthropic tool specification
        """
        return {
            "name": self.tool_definition.name,
            "description": self.tool_definition.description,
            "input_schema": self.tool_definition.input_schema
        }
    
    def to_mistral_tool(self) -> Dict[str, Any]:
        """
        Convert the MCP tool to a Mistral tool format.
        
        Returns:
            A Mistral tool specification
        """
        from mistral_common.protocol.instruct.tool_calls import Tool, Function
        
        return Tool(
            function=Function(
                name=self.tool_definition.name,
                description=self.tool_definition.description,
                parameters=self.tool_definition.input_schema,
            )
        )
    
    def to_nous_hermes_pro_tool(self) -> Dict[str, Any]:
        """
        Convert the MCP tool to a Nous Hermes Pro tool format.
        
        Returns:
            A Nous Hermes Pro tool specification
        """
        properties = {}
        for prop_name, prop_info in self.tool_definition.input_schema.get("properties", {}).items():
            properties[prop_name] = {
                "type": prop_info.get("type", "string"),
                "description": prop_info.get("description", "")
            }
            
            # Handle enum types
            if "enum" in prop_info:
                properties[prop_name]["enum"] = prop_info["enum"]
            
            # Handle array types
            if prop_info.get("type") == "array" and "items" in prop_info:
                properties[prop_name]["items"] = {
                    "type": prop_info["items"].get("type", "string")
                }
                if "enum" in prop_info["items"]:
                    properties[prop_name]["items"]["enum"] = prop_info["items"]["enum"]
        
        return {
            "type": "function",
            "function": {
                "name": self.tool_definition.name,
                "description": self.tool_definition.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": self.tool_definition.input_schema.get("required", [])
                }
            }
        }


class MCPToolRegistry:
    """
    Registry for discovering and managing Model Context Protocol tools.
    
    Allows for automatic discovery of tools from MCP servers and integrates
    them into the ToolAgents framework.
    """
    
    @staticmethod
    def discover_tools_from_server(server_url: str, debug_mode: bool = False) -> List[MCPTool]:
        """
        Discover tools available from an MCP server.
        
        Args:
            server_url: URL of the MCP server
            debug_mode: Enable debug mode for discovered tools
            
        Returns:
            List of MCPTool instances
        """
        try:
            # Query the server for available tools
            response = requests.get(
                f"{server_url}/tools",
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            
            tools_data = response.json()
            tools = []
            
            # Create MCPTool instances for each discovered tool
            for tool_data in tools_data:
                if "name" in tool_data and "description" in tool_data and "input_schema" in tool_data:
                    # Create the tool definition
                    tool_def = MCPToolDefinition(
                        name=tool_data["name"],
                        description=tool_data["description"],
                        input_schema=tool_data["input_schema"],
                        server_url=server_url
                    )
                    
                    # Create the MCP tool
                    tool = MCPTool(tool_definition=tool_def, debug_mode=debug_mode)
                    tools.append(tool)
            
            return tools
            
        except Exception as e:
            print(f"Error discovering tools from MCP server at {server_url}: {str(e)}")
            return []
    
    @staticmethod
    def add_mcp_tools_to_registry(
        tool_registry: 'ToolRegistry',
        mcp_tools: List[MCPTool]
    ) -> None:
        """
        Add MCP tools to a ToolRegistry.
        
        Args:
            tool_registry: The ToolRegistry to add tools to
            mcp_tools: List of MCPTool instances to add
        """
        for tool in mcp_tools:
            # Create a wrapper function that calls the execute method of the MCPTool
            def create_execute_wrapper(t):
                def execute_wrapper(**kwargs):
                    return t.execute(kwargs)
                
                # Set the wrapper function's name to match the tool name
                execute_wrapper.__name__ = t.tool_definition.name
                
                # Set a docstring based on the tool description
                execute_wrapper.__doc__ = t.tool_definition.description
                
                return execute_wrapper
            
            # Create the wrapper function
            wrapper_func = create_execute_wrapper(tool)
            
            # Create a FunctionTool from the OpenAI format and the wrapper function
            function_tool = FunctionTool.from_openai_tool(
                tool.to_openai_tool(),
                wrapper_func
            )
            
            # Add the tool to the registry
            tool_registry.add_tool(function_tool)