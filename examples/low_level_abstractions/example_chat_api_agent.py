import datetime
import json
import os
import uuid

from ToolAgents import ToolRegistry
from ToolAgents.agents import ChatToolAgent
from ToolAgents.messages.chat_message import ChatMessage, ChatMessageRole, TextContent
from ToolAgents.provider import AnthropicChatAPI, AnthropicSettings, OpenAIChatAPI, OpenAISettings, \
    LlamaCppServerProvider
from ToolAgents.provider.chat_api_provider.groq import GroqChatAPI, GroqSettings
from ToolAgents.provider.chat_api_provider.mistral import MistralChatAPI, MistralSettings
from ToolAgents.provider.generation_provider.default_implementations import HuggingFaceTokenizer, MistralTokenizer
from ToolAgents.provider.generation_provider.llama_cpp_server import LlamaCppSamplingSettings

from example_tools import calculator_function_tool, current_datetime_function_tool, get_weather_function_tool

from dotenv import load_dotenv

load_dotenv()

# Local OpenAI like API, like vllm or llama-cpp-server
# api = OpenAIChatAPI(api_key="token-abc123", base_url="http://127.0.0.1:8080/v1", model="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")
# settings = OpenAISettings()

# Official OpenAI API
#api = OpenAIChatAPI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")
#settings = OpenAISettings()

# Anthropic API
#api = AnthropicChatAPI(api_key=os.getenv("ANTHROPIC_API_KEY"), model="claude-3-5-sonnet-20241022")
#settings = AnthropicSettings()

# Groq API
#api = GroqChatAPI(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.3-70b-versatile")
#settings = GroqSettings()

# Llama Cpp Generation Server API
api = LlamaCppServerProvider("http://127.0.0.1:8080")
settings = LlamaCppSamplingSettings()

# Mistral API
#api = MistralChatAPI(api_key=os.getenv("MISTRAL_API_KEY"), model="mistral-small-latest")
#settings = MistralSettings()

# Create the ChatAPIAgent
agent = ChatToolAgent(chat_api=api)

# Set sampling settings
settings.temperature = 0.45
settings.top_p = 1.0

# Define the tools
tools = [calculator_function_tool, current_datetime_function_tool, get_weather_function_tool]
tool_registry = ToolRegistry()

tool_registry.add_tools(tools)

messages = [
    ChatMessage.create_system_message("You are a helpful assistant with tool calling capabilities. Only reply with a tool call if the function exists in the library provided by the user. Use JSON format to output your function calls. If it doesn't exist, just reply directly in natural language. When you receive a tool call response, use the output to format an answer to the original user question."),
    ChatMessage.create_user_message("Retrieve the date and time in the format: %Y-%m-%d %H:%M:%S.")
]


chat_response = agent.get_response(
    messages=messages,
    settings=settings, tool_registry=tool_registry)

print(chat_response.response)

