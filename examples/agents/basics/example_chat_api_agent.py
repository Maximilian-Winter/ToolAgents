import json
import os
from copy import copy

from ToolAgents.function_tool import ToolRegistry
from ToolAgents.agents import ChatToolAgent
from ToolAgents.data_models.messages import ChatMessage
from ToolAgents.provider import (
    AnthropicChatAPI,
    OpenAIChatAPI,
    GroqChatAPI,
    MistralChatAPI,
    CompletionProvider,
)
from ToolAgents.provider.completion_provider.default_implementations import (
    LlamaCppServer,
)
from ToolAgents.provider.llm_provider import LLMSetting, SettingLevel

from example_tools import (
    calculator_function_tool,
    current_datetime_function_tool,
    get_weather_function_tool,
)

from dotenv import load_dotenv

load_dotenv()

# Local OpenAI like API, like vllm or llama-cpp-server with jinja flag for tool calling.
#api = OpenAIChatAPI(api_key="token-abc123", base_url="http://127.0.0.1:8080/v1", model="Mistral-Small-3.2-24B-Instruct-2506")

# Official OpenAI API
#api = OpenAIChatAPI(api_key=os.getenv("OPENAI_API_KEY"), model="google/gemini-3-flash-preview")

# Openrouter API
api = OpenAIChatAPI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="qwen/qwen3.5-9b",
    base_url="https://openrouter.ai/api/v1",
)

# Anthropic API
#api = OpenAIChatAPI(api_key=os.getenv("GOOGLE_API_KEY"), base_url="https://generativelanguage.googleapis.com/v1beta/openai/", model="gemini-2.0-flash-lite-preview-02-05")


# Groq API
#api = GroqChatAPI(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.3-70b-versatile")


# Mistral API
#api = MistralChatAPI(api_key=os.getenv("MISTRAL_API_KEY"), model="mistral-small-latest")

# Create the ChatAPIAgent
agent = ChatToolAgent(chat_api=api)

# Create a samplings settings object
settings = api.get_default_settings()

# Set sampling settings
settings.temperature = 0.3
settings.top_p = 0.9

# Add settings



# Define the tools
tools = [
    calculator_function_tool,
    current_datetime_function_tool,
    get_weather_function_tool,
]
tool_registry = ToolRegistry()

tool_registry.add_tools(tools)

messages = [
    ChatMessage.create_user_message(
        "Perform all the following tasks: Get the current weather in celsius in the city of London, Great Britain, New York City, New York and at the North Pole, Arctica. Solve the following calculations: 42 * 42, 74 + 26, 7 * 26, 4 + 6  and 96/8. And retrieve the date and time in the format: %Y-%m-%d %H:%M:%S."
    ),
]

print(f"Prompt: {messages[0].get_text_content()}")

chat_response = agent.get_response(
    messages=copy(messages), tool_registry=tool_registry, settings=settings
)

print(f"Response: {chat_response.response}")

print(f"All messages: {'\n'.join([msg.model_dump_json(indent=2) for msg in chat_response.messages])}")



