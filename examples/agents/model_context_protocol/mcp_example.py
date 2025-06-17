import os
import shutil
from copy import copy

import asyncio
from time import sleep

from mcp import StdioServerParameters


from ToolAgents import ToolRegistry
from ToolAgents.agents import ChatToolAgent
from ToolAgents.data_models.messages import ChatMessage
from ToolAgents.provider import (
    MistralChatAPI, OpenAIChatAPI
)


from dotenv import load_dotenv

from ToolAgents.utilities.mcp_session import MCPServerTools

load_dotenv()


# Mistral API
api = OpenAIChatAPI(api_key="token-abc123", base_url="http://127.0.0.1:8080/v1", model="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")

# Create the ChatAPIAgent
agent = ChatToolAgent(chat_api=api)

# Create a samplings settings object
settings = api.get_default_settings()

# Set sampling settings
settings.temperature = 0.45
settings.top_p = 1.0


#npx_path = shutil.which("npx")
#server_params = StdioServerParameters(
#    command=npx_path,
#    args=[
#        "-y",
#        "@modelcontextprotocol/server-filesystem",
#        r"H:\LLM_Dev\rap"
#    ]
#)
#mcp_server_tools = MCPServerTools()
#loop = asyncio.new_event_loop()
#asyncio.set_event_loop(loop)
#tools = loop.run_until_complete(mcp_server_tools.load_from_stdio_server(server_params=server_params))
#while loop.is_running():
#    sleep(0.5)
mcp_server_tools = MCPServerTools()
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
tools = loop.run_until_complete(mcp_server_tools.load_from_http_server(server_kwargs={"url": "http://127.0.0.1:8042/mcp"}))
while loop.is_running():
    sleep(0.5)


tool_registry = ToolRegistry()

tool_registry.add_tools(tools)

messages = [
    ChatMessage.create_system_message(
        "You are a helpful assistant with tool calling capabilities. Only reply with a tool call if the function exists in the library provided by the user. Use JSON format to output your function calls. If it doesn't exist, just reply directly in natural language. When you receive a tool call response, use the output to format an answer to the original user question."
    ),
    ChatMessage.create_user_message(
        #"Write a poem about Donald Trump and Joe Biden into a txt file called 'hope.txt' in the 'H:\\LLM_Dev\\rap' directory",
        "Use the tool 'Greet' to greet me with my name 'John'.",
    ),
]


chat_response = agent.get_response(
    messages=copy(messages), settings=settings, tool_registry=tool_registry
)
print(chat_response.response)

if loop.is_running():
    loop.stop()
    while loop.is_running():
        sleep(0.1)
        loop.stop()
    loop.close()
print("Finished", flush=True)
