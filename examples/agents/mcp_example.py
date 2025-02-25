import os
from typing import Dict, Any

from ToolAgents import ToolRegistry
from ToolAgents.agents import ChatToolAgent
from ToolAgents.messages import ChatHistory
from ToolAgents.messages.chat_message import ChatMessage
from ToolAgents.provider import OpenAIChatAPI
from ToolAgents.mcp_tool import MCPToolRegistry, MCPToolDefinition, MCPTool

from dotenv import load_dotenv

load_dotenv()

# Set up API client
api = OpenAIChatAPI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")
settings = api.get_default_settings()
settings.temperature = 0.2
agent = ChatToolAgent(chat_api=api)

# Create a tool registry
tool_registry = ToolRegistry()

# Example 1: Manually define MCP tools
def setup_manual_mcp_tools():
    """
    Example of manually defining MCP tools
    """
    # Weather tool definition
    weather_tool_def = MCPToolDefinition(
        name="get_weather",
        description="Get the current weather for a location",
        input_schema={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string", 
                    "description": "The city and state/country, e.g., 'San Francisco, CA'"
                },
                "unit": {
                    "type": "string",
                    "description": "The temperature unit",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["location"]
        },
        server_url="http://localhost:8000/mcp"
    )
    
    # Create the MCP tool
    weather_tool = MCPTool(
        tool_definition=weather_tool_def,
        debug_mode=True
    )
    
    # Create a calculator tool definition
    calculator_tool_def = MCPToolDefinition(
        name="calculate",
        description="Perform a mathematical calculation",
        input_schema={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate, e.g., '2 + 2'"
                }
            },
            "required": ["expression"]
        },
        server_url="http://localhost:8000/mcp"
    )
    
    # Create the MCP tool
    calculator_tool = MCPTool(
        tool_definition=calculator_tool_def,
        debug_mode=True
    )
    
    # Add these tools to a ToolRegistry
    MCPToolRegistry.add_mcp_tools_to_registry(tool_registry, [weather_tool, calculator_tool])
    
    return [weather_tool, calculator_tool]

# Example 2: Automatically discover tools from an MCP server
def discover_mcp_tools(server_url: str = "http://localhost:8000/mcp"):
    """
    Example of automatically discovering tools from an MCP server
    """
    print(f"Discovering MCP tools from server: {server_url}")
    
    # Discover tools
    tools = MCPToolRegistry.discover_tools_from_server(server_url, debug_mode=True)
    
    print(f"Discovered {len(tools)} tools:")
    for tool in tools:
        print(f"- {tool.tool_definition.name}: {tool.tool_definition.description}")
    
    # Add discovered tools to the tool registry
    MCPToolRegistry.add_mcp_tools_to_registry(tool_registry, tools)
    
    return tools

# For this example, we'll use the manually defined tools
# In a real application, you might use the discovery method instead
mcp_tools = setup_manual_mcp_tools()
# Uncomment to use discovery instead:
# mcp_tools = discover_mcp_tools()

# Initialize chat history
chat_history = ChatHistory()
chat_history.add_system_message("""
You are an assistant with access to various tools. When a user asks for information that requires using
a tool, use the appropriate tool to get the information and respond based on the result.

Available tools:
- get_weather: Get the current weather for a location
- calculate: Perform a mathematical calculation

Please help the user with their requests by using these tools when appropriate.
""")

def process_user_input(user_input: str) -> str:
    """Process user input and get a response from the agent"""
    chat_history.add_user_message(user_input)
    
    response = agent.get_response(
        messages=chat_history.get_messages(),
        settings=settings,
        tool_registry=tool_registry
    )
    
    chat_history.add_messages(response.messages)
    return response.response

# Example usage
if __name__ == "__main__":
    print("Model Context Protocol (MCP) Tool Example")
    print("----------------------------------------")
    print("This example shows how to use Model Context Protocol tools with ToolAgents.")
    print("Available commands:")
    print("- Type 'exit' to quit")
    print("- Type 'tools' to see available tools")
    print("- Otherwise, your input will be processed by the agent\n")
    
    while True:
        user_input = input("User > ")
        if user_input.lower() == "exit":
            break
        elif user_input.lower() == "tools":
            print("\nAvailable tools:")
            for tool in mcp_tools:
                print(f"- {tool.tool_definition.name}: {tool.tool_definition.description}")
            print()
        else:
            # In a real application, these requests would be forwarded to a real MCP server
            # For this example, we'll use a mock response
            print("\nNote: This is an example without a real MCP server connection.")
            print("In a real application, requests would be sent to the MCP server.\n")
            
            print("Assistant > ", end="")
            # This would normally come from the agent, but we'll mock it for this example
            if "weather" in user_input.lower():
                print("I'll check the weather for you.")
                print("Using the get_weather tool...")
                print("The current weather in the location you asked about is 72°F and sunny.")
            elif any(term in user_input.lower() for term in ["calculate", "math", "compute", "+"]):
                print("I'll calculate that for you.")
                print("Using the calculate tool...")
                if "2+2" in user_input or "2 + 2" in user_input:
                    print("The result is 4.")
                else:
                    print("The result of your calculation is 42.")
            else:
                print("I don't have specific tools to help with that request, but I can answer general questions.")
            
            print()