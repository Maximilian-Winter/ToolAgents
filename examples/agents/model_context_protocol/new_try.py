import os
import shutil
import sys
from copy import copy

import asyncio
from time import sleep

from mcp import StdioServerParameters

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


from dotenv import load_dotenv

from ToolAgents.utilities.mcp_session import MCPServerTools

load_dotenv()

#api = CompletionProvider(completion_endpoint=LlamaCppServer("http://127.0.0.1:8080"))
# Official OpenAI API
#api = OpenAIChatAPI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")


# Anthropic API
# api = OpenAIChatAPI(api_key=os.getenv("GOOGLE_API_KEY"), base_url="https://generativelanguage.googleapis.com/v1beta/openai/", model="gemini-2.0-flash-lite-preview-02-05")


# Groq API
# api = GroqChatAPI(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.3-70b-versatile")

# Llama Cpp Server Completion Based API with Mistral model
# api = CompletionProvider(completion_endpoint=LlamaCppServer("http://127.0.0.1:8080"))

# Mistral API
api = MistralChatAPI(api_key=os.getenv("MISTRAL_API_KEY"), model="mistral-small-latest")

# Create the ChatAPIAgent
agent = ChatToolAgent(chat_api=api)

# Create a samplings settings object
settings = api.get_default_settings()

# Set sampling settings
settings.temperature = 0.45
settings.top_p = 1.0


npx_path = shutil.which("npx")
server_params = StdioServerParameters(
    command=npx_path,
    args=[
        "-y",
        "@modelcontextprotocol/server-filesystem",
        r"H:\LLM_Dev\rap"
    ]
)
mcp_server_tools = MCPServerTools()

loop = asyncio.new_event_loop()
tools = loop.run_until_complete(mcp_server_tools.load_from_stdio_server(server_params=server_params))
loop.stop()
while loop.is_running():
    sleep(0.1)
    loop.stop()
loop.close()
loop = None
tool_registry = ToolRegistry()

tool_registry.add_tools(tools)

messages = [
    ChatMessage.create_system_message(
        "You are a helpful assistant with tool calling capabilities. Only reply with a tool call if the function exists in the library provided by the user. Use JSON format to output your function calls. If it doesn't exist, just reply directly in natural language. When you receive a tool call response, use the output to format an answer to the original user question."
    ),
    ChatMessage.create_user_message(
        "Write a poem about Donald Trump and Joe Biden into a txt file called 'hope.txt' in the 'H:\\LLM_Dev\\rap' directory",
    ),
]


chat_response = agent.get_response(
    messages=copy(messages), settings=settings, tool_registry=tool_registry
)
print(chat_response.response)


print("Finished", flush=True)
