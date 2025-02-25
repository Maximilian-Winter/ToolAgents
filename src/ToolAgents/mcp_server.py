import json
import logging
from typing import List, Dict, Any, Optional, Union, Callable

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import uvicorn

from ToolAgents import FunctionTool, ToolRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class MCPRequestParams(BaseModel):
    """Parameters for an MCP tool request"""
    parameters: Dict[str, Any] = Field(..., description="Parameters for the tool")


class MCPResponseSuccess(BaseModel):
    """Successful response from an MCP tool"""
    result: Any = Field(..., description="Result of the tool execution")


class MCPResponseError(BaseModel):
    """Error response from an MCP tool"""
    error: str = Field(..., description="Error message")
    code: int = Field(500, description="Error code")


class MCPToolInfo(BaseModel):
    """Information about an MCP tool"""
    name: str = Field(..., description="The name of the tool")
    description: str = Field(..., description="Description of what the tool does")
    input_schema: Dict[str, Any] = Field(..., description="JSON schema describing the expected input format")


class MCPServer:
    """
    A server that provides Model Context Protocol (MCP) services based on FunctionTools.
    
    This server exposes tools as MCP-compatible API endpoints that can be called by
    MCP clients or integrated into other systems.
    
    Args:
        tool_registry: A ToolRegistry containing the tools to expose
        host: Host address to run the server on (default: "0.0.0.0")
        port: Port to run the server on (default: 8000)
        prefix: API prefix for the MCP endpoints (default: "/mcp")
        debug_mode: Enable debug mode (default: False)
    """
    
    def __init__(
        self,
        tool_registry: Union[ToolRegistry, List[FunctionTool]],
        host: str = "0.0.0.0",
        port: int = 8000,
        prefix: str = "/mcp",
        debug_mode: bool = False
    ):
        # Convert a list of tools to a ToolRegistry if necessary
        if isinstance(tool_registry, list):
            self.tool_registry = ToolRegistry()
            self.tool_registry.add_tools(tool_registry)
        else:
            self.tool_registry = tool_registry
            
        self.host = host
        self.port = port
        self.prefix = prefix.rstrip("/")  # Remove trailing slash if present
        self.debug_mode = debug_mode
        
        # Create FastAPI app
        self.app = FastAPI(
            title="MCP Tool Server",
            description="Model Context Protocol server for executing tool functions",
            version="1.0.0"
        )
        
        # Register routes
        self._register_routes()
    
    def _register_routes(self):
        """Register FastAPI routes for MCP endpoints"""
        
        @self.app.get(f"{self.prefix}/tools")
        async def list_tools():
            """List all available tools"""
            tools = []
            for tool_name, tool in self.tool_registry.tools.items():
                openai_schema = tool.to_openai_tool()
                
                tool_info = MCPToolInfo(
                    name=openai_schema["function"]["name"],
                    description=openai_schema["function"]["description"],
                    input_schema=openai_schema["function"]["parameters"]
                )
                
                tools.append(tool_info.model_dump())
            
            return tools
        
        @self.app.get(f"{self.prefix}/tools/{{tool_name}}")
        async def get_tool_info(tool_name: str):
            """Get information about a specific tool"""
            if tool_name not in self.tool_registry.tools:
                raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
            
            tool = self.tool_registry.tools[tool_name]
            openai_schema = tool.to_openai_tool()
            
            tool_info = MCPToolInfo(
                name=openai_schema["function"]["name"],
                description=openai_schema["function"]["description"],
                input_schema=openai_schema["function"]["parameters"]
            )
            
            return tool_info.model_dump()
        
        @self.app.post(f"{self.prefix}/tools/{{tool_name}}")
        async def execute_tool(tool_name: str, request: MCPRequestParams):
            """Execute a tool with the provided parameters"""
            if tool_name not in self.tool_registry.tools:
                raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
            
            tool = self.tool_registry.tools[tool_name]
            
            try:
                if self.debug_mode:
                    logger.info(f"Executing tool '{tool_name}' with parameters: {request.parameters}")
                
                # Execute the tool
                result = tool.execute(request.parameters)
                
                if self.debug_mode:
                    logger.info(f"Tool '{tool_name}' result: {result}")
                
                return MCPResponseSuccess(result=result).model_dump()
                
            except Exception as e:
                error_message = str(e)
                logger.error(f"Error executing tool '{tool_name}': {error_message}")
                
                raise HTTPException(
                    status_code=500,
                    detail=MCPResponseError(
                        error=error_message,
                        code=500
                    ).model_dump()
                )
    
    def run(self):
        """Run the MCP server"""
        logger.info(f"Starting MCP server on {self.host}:{self.port}")
        uvicorn.run(self.app, host=self.host, port=self.port)
    
    def run_in_thread(self):
        """Run the MCP server in a separate thread"""
        import threading
        
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()
        return thread


class MCPServerConfig(BaseModel):
    """Configuration for an MCP server"""
    host: str = Field("0.0.0.0", description="Host address to run the server on")
    port: int = Field(8000, description="Port to run the server on")
    prefix: str = Field("/mcp", description="API prefix for the MCP endpoints")
    debug_mode: bool = Field(False, description="Enable debug mode")


def create_and_run_mcp_server(
    tools: Union[ToolRegistry, List[FunctionTool]],
    config: Optional[MCPServerConfig] = None
) -> MCPServer:
    """
    Create and run an MCP server with the provided tools.
    
    Args:
        tools: A ToolRegistry or list of FunctionTools to expose
        config: Optional server configuration
        
    Returns:
        The running MCP server instance
    """
    if config is None:
        config = MCPServerConfig()
    
    server = MCPServer(
        tool_registry=tools,
        host=config.host,
        port=config.port,
        prefix=config.prefix,
        debug_mode=config.debug_mode
    )
    
    # Run the server in a background thread
    server.run_in_thread()
    
    return server