

import json
import os
from copy import copy

from ToolAgents import ToolRegistry
from ToolAgents.agents import ChatToolAgent
from ToolAgents.data_models.messages import ChatMessage
from ToolAgents.provider import (
    AnthropicChatAPI,
    GoogleGenAIChatAPI,
    OpenAIChatAPI,
    GroqChatAPI,
    MistralChatAPI,
    CompletionProvider,
)
from ToolAgents.provider.completion_provider.default_implementations import (
    LlamaCppServer,
)


from dotenv import load_dotenv

from ToolAgents.utilities.mcp_session import MCPServerTools

load_dotenv()

#api = CompletionProvider(completion_endpoint=LlamaCppServer("http://127.0.0.1:8080"))
# Official OpenAI API
# api = OpenAIChatAPI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

# Openrouter API
api = OpenAIChatAPI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="meta-llama/llama-3.3-70b-instruct",
    base_url="https://openrouter.ai/api/v1",
)

# Anthropic API
# api = OpenAIChatAPI(api_key=os.getenv("GOOGLE_API_KEY"), base_url="https://generativelanguage.googleapis.com/v1beta/openai/", model="gemini-2.0-flash-lite-preview-02-05")


# Groq API
# api = GroqChatAPI(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.3-70b-versatile")

# Llama Cpp Server Completion Based API with Mistral model
# api = CompletionProvider(completion_endpoint=LlamaCppServer("http://127.0.0.1:8080"))

# Mistral API
# api = MistralChatAPI(api_key=os.getenv("MISTRAL_API_KEY"), model="mistral-small-latest")

# Create the ChatAPIAgent
agent = ChatToolAgent(chat_api=api)

# Create a samplings settings object
settings = api.get_default_settings()

# Set sampling settings
settings.temperature = 0.45
settings.top_p = 1.0
from mcp import StdioServerParameters

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
mcp_server_tools = MCPServerTools()

mcp_server_tools.load_from_stdio_server(server_params=server_params)
# Define the tools
tools = mcp_server_tools.tools
tool_registry = ToolRegistry()

tool_registry.add_tools(tools)

messages = [
    ChatMessage.create_system_message(
        "You are a helpful assistant with tool calling capabilities. Only reply with a tool call if the function exists in the library provided by the user. Use JSON format to output your function calls. If it doesn't exist, just reply directly in natural language. When you receive a tool call response, use the output to format an answer to the original user question."
    ),
    ChatMessage.create_user_message(
        "Calculate 420 x 420 using your calculator tool."
    ),
]


chat_response = agent.get_response(
    messages=copy(messages), settings=settings, tool_registry=tool_registry
)

print(chat_response.response)
