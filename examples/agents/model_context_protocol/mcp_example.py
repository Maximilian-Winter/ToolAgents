import os
import shutil
from typing import Dict, Any

from ToolAgents import ToolRegistry
from fuck import get_mcp_function_tools
from ToolAgents.agents import ChatToolAgent
from ToolAgents.data_models.chat_history import ChatHistory
from ToolAgents.data_models.messages import ChatMessage
from ToolAgents.provider import OpenAIChatAPI
from ToolAgents.model_context_protocol.mcp_tool import MCPToolRegistry, MCPToolDefinition, MCPTool

from dotenv import load_dotenv

load_dotenv()

# Set up API client
api = OpenAIChatAPI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")
settings = api.get_default_settings()
settings.temperature = 0.2
agent = ChatToolAgent(chat_api=api)


npx_path = shutil.which("npx")
# Get filesystem tools from MCP server
filesystem_tools = get_mcp_function_tools([
    npx_path, "-y", "@modelcontextprotocol/server-filesystem",
    "H:\\LLM_Dev\\rap"
])

# Add to your existing registry
registry = ToolRegistry()
registry.add_tools(filesystem_tools)

# Initialize chat history
chat_history = ChatHistory()
chat_history.add_system_message(
    """
You are an assistant with access to various tools. When a user asks for information that requires using
a tool, use the appropriate tool to get the information and respond based on the result.

Please help the user with their requests by using these tools when appropriate.
"""
)


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
            for tool in registry.get_tools():
                print(
                    f"- {tool.model.__name__}: {tool.model.model_json_schema()}"
                )
            print()
        else:
            chat_history.add_user_message(user_input)
            print("Assistant > ", end="")
            response = agent.get_response(settings=settings, messages=chat_history.get_messages(), tool_registry=registry)
            print(response.response)
            chat_history.add_messages(response.messages)
            print()
