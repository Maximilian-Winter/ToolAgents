import datetime
import json
import os
import uuid

from ToolAgents import ToolRegistry
from ToolAgents.agents import ChatToolAgent
from ToolAgents.messages.chat_message import ChatMessage, ChatMessageRole, TextContent
from ToolAgents.provider import AnthropicChatAPI, AnthropicSettings, OpenAIChatAPI, OpenAISettings
from ToolAgents.provider.chat_api_provider.groq import GroqChatAPI, GroqSettings
from ToolAgents.provider.chat_api_provider.mistral import MistralChatAPI, MistralSettings

from example_tools import calculator_function_tool, current_datetime_function_tool, get_weather_function_tool

from dotenv import load_dotenv

load_dotenv()

api = OpenAIChatAPI(api_key="token-abc123", base_url="http://127.0.0.1:8000/v1", model="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")
settings = OpenAISettings()

#api = OpenAIChatAPI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://api.openai.com/v1", model="gpt-4o-mini")
#settings = OpenAISettings()

#api = AnthropicChatAPI(api_key=os.getenv("ANTHROPIC_API_KEY"), model="claude-3-5-sonnet-20241022")
#settings = AnthropicSettings()

#api = GroqChatAPI(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.3-70b-versatile")
#settings = GroqSettings()

# api = MistralChatAPI(api_key=os.getenv("MISTRAL_API_KEY"), model="mistral-small-latest")
# settings = MistralSettings()

# Create the ChatAPIAgent
agent = ChatToolAgent(chat_api=api, debug_output=True)

settings.temperature = 0.45
settings.top_p = 0.85

# Define the tools
tools = [calculator_function_tool, current_datetime_function_tool, get_weather_function_tool]
tool_registry = ToolRegistry()

tool_registry.add_tools(tools)
messages = [
  {
    "role": "system",
    "content": "You are Funky, an AI assistant specialized in interpreting user requests and generating appropriate function calls. Only you can see the tool results, once you get the results, you have to write a response to the user with the requested information."
  },
  {
    "role": "user",
    "content": "Perform all the following tasks: Get the current weather in celsius in the city of London, Great Britain, New York City, New York and at the North Pole, Arctica. Solve the following calculations: 42 * 42, 74 + 26, 7 * 26, 4 + 6  and 96/8. And retrieve the date and time in the format: %Y-%m-%d %H:%M:%S."
  }
]


result = agent.get_response(
    messages=ChatMessage.convert_list_of_dicts(messages),
    settings=settings, tool_registry=tool_registry)

print(result.model_dump_json(indent=4))
print('\n'.join([msg.model_dump_json(indent=4) for msg in agent.last_messages_buffer]))

result = agent.get_streaming_response(
    messages=ChatMessage.convert_list_of_dicts(messages),
    settings=settings, tool_registry=tool_registry)

print(result.model_dump_json(indent=4))
print('\n'.join([msg.model_dump_json(indent=4) for msg in agent.last_messages_buffer]))
